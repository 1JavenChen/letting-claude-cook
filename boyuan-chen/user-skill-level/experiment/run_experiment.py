"""Run the user-model adaptation experiments.

Two suites share one collection pipeline (pick with --suite):
  education (default) - 30 questions x (4 education levels x 2 channels + control)
                        x N reps -> out/responses.jsonl  (v1 study)
  skill               - 24 coding tasks x (4 skill levels x 2 channels + control)
                        x N reps -> out/responses_skill.jsonl  (v2 study; tasks
                        from tasks.json, conditions from conditions_skill.json)

Task/question text is byte-identical across conditions; only the credential
sentence in the system prompt or message prefix varies. Execution order is
shuffled with a recorded seed so rate-limit drift cannot alias with conditions.

Four backends:
  groq (default)  - Groq free tier (GROQ_API_KEY from console.groq.com/keys).
                    Free, fast; daily token limits -> rerun to resume if hit.
  local           - any OpenAI-compatible server (Ollama, LM Studio) at
                    LOCAL_API_BASE (default http://localhost:11434/v1).
                    Fully free, no account.
  hf              - Hugging Face Inference Providers (HF_TOKEN); needs
                    inference credits on the account.
  anthropic       - Message Batches API (50% price). Needs ANTHROPIC_API_KEY
                    (and ANTHROPIC_WORKSPACE_ID for user-scoped keys).

The sequential backends (groq, local, hf) checkpoint each success to
out/responses.jsonl; rerun the same command to resume. Failures go to
out/errors.log and are retried on the next run; 5 consecutive failures abort.

Usage:
    python run_experiment.py --dry-run
    python run_experiment.py                          # education suite, groq
    python run_experiment.py --suite skill --dry-run
    python run_experiment.py --suite skill            # 648 requests
    python run_experiment.py --suite skill --reps 1   # third-size pilot
    python run_experiment.py --backend local
    python run_experiment.py --backend anthropic --resume msgbatch_xxx
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-20b",
    "local": "qwen3:4b-instruct-2507",
    "hf": "Qwen/Qwen3-4B-Instruct-2507",
    "anthropic": "claude-opus-5-5",
}
ANTHROPIC_MAX_TOKENS = 8000  # headroom for thinking tokens + a long answer
HF_MAX_TOKENS = 1600

# Each suite: its stimulus file, condition file, output file, and completion
# budget (code answers need more room than prose answers).
SUITES = {
    "education": {"out": "responses.jsonl", "max_tokens": 1600},
    "skill": {"out": "responses_skill.jsonl", "max_tokens": 2400},
    # v2.1: harder Codeforces battery (tasks_cf.json), same persona conditions
    "cf": {"out": "responses_cf.jsonl", "max_tokens": 3000},
}
TASK_FILES = {"skill": "tasks.json", "cf": "tasks_cf.json"}
TASK_FILE_OVERRIDE = None  # set by --tasks-file
TEMPERATURE = 0.7  # fixed sampling param, recorded per response
LOCAL_API_BASE = os.environ.get("LOCAL_API_BASE", "http://localhost:11434/v1")
GROQ_API_BASE = "https://api.groq.com/openai/v1"


def build_cells(suite="education"):
    """Yield (custom_id_prefix, metadata, system, user_message) for every cell."""
    if suite in TASK_FILES:
        yield from build_cells_skill(suite)
        return
    questions = json.loads((HERE / "questions.json").read_text())
    cond = json.loads((HERE / "conditions.json").read_text())

    for q in questions:
        meta_base = {"question_id": q["id"], "domain": q["domain"], "question_type": q["type"]}

        c = cond["control"]
        yield (
            f"{q['id']}_control_none",
            {**meta_base, "education_level": "control", "channel": "none"},
            c["system_template"],
            c["message_template"].format(question=q["text"]),
        )

        for level_key, level_desc in cond["education_levels"].items():
            for channel_key, ch in cond["channels"].items():
                yield (
                    f"{q['id']}_{level_key}_{channel_key}",
                    {**meta_base, "education_level": level_key, "channel": channel_key},
                    ch["system_template"].format(level=level_desc),
                    ch["message_template"].format(level=level_desc, question=q["text"]),
                )


def build_cells_skill(suite="skill"):
    """Cells for a coding suite: N tasks x (4x2 + control) = 9N.

    'skill' uses tasks.json (24 function-level tasks); 'cf' uses tasks_cf.json
    (Codeforces stdin/stdout problems). Persona wording is shared.
    """
    tasks = json.loads((HERE / (TASK_FILE_OVERRIDE or TASK_FILES[suite])).read_text())
    cond = json.loads((HERE / "conditions_skill.json").read_text())

    for t in tasks:
        meta_base = {
            "task_id": t["id"],
            "language": t["language"],
            "difficulty": t["difficulty"],
            "hardening": t["hardening"],
            "entry_point": t["entry_point"],
        }

        c = cond["control"]
        yield (
            f"{t['id']}_control_none",
            {**meta_base, "skill_level": "control", "channel": "none"},
            c["system_template"],
            c["message_template"].format(task=t["prompt"]),
        )

        for level_key, level_desc in cond["skill_levels"].items():
            for channel_key, ch in cond["channels"].items():
                yield (
                    f"{t['id']}_{level_key}_{channel_key}",
                    {**meta_base, "skill_level": level_key, "channel": channel_key},
                    ch["system_template"].format(level=level_desc),
                    ch["message_template"].format(level=level_desc, task=t["prompt"]),
                )


# ------------------------------------------------------------- HF backend


def make_hf_client():
    from huggingface_hub import InferenceClient

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "Set HF_TOKEN to a Hugging Face access token "
            "(free: https://huggingface.co/settings/tokens, 'read' scope)."
        )
    return InferenceClient(token=token)


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def hf_chat(client, model, system, user_message, max_tokens=HF_MAX_TOKENS,
            temperature=0.7, retries=6):
    """One chat completion with exponential backoff on rate limits / 5xx.

    Client errors other than 429 (bad model, bad request) fail immediately.
    """
    for attempt in range(retries):
        try:
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = resp.choices[0]
            usage = getattr(resp, "usage", None)
            # Defensive: some models emit <think> reasoning blocks; keep answers only.
            text = THINK_RE.sub("", choice.message.content or "")
            return {
                "response_text": text,
                "stop_reason": choice.finish_reason,
                "output_tokens": getattr(usage, "completion_tokens", None),
                "input_tokens": getattr(usage, "prompt_tokens", None),
            }
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            if attempt == retries - 1:
                raise
            delay = min(5 * 2**attempt + random.uniform(0, 3), 180)
            print(f"    retry {attempt + 1}/{retries - 1} in {delay:.0f}s: {type(e).__name__}: {e}")
            time.sleep(delay)


# ---------------------------------------------------------- local backend


if os.environ.get("FORCE_IPV4"):
    # Some networks black-hole IPv6 to the API host; urllib tries IPv6 first and
    # only falls back after the full socket timeout. Resolve IPv4 only.
    import socket as _socket
    _orig_getaddrinfo = _socket.getaddrinfo

    def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return _orig_getaddrinfo(host, port, _socket.AF_INET, type, proto, flags)

    _socket.getaddrinfo = _ipv4_only_getaddrinfo


def _ssl_context():
    import ssl

    try:
        # python.org macOS builds don't see the system root certs; use certifi's.
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


# Cloudflare blocks urllib's default User-Agent (error 1010).
HTTP_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "education-level-experiment/1.0",
}


def openai_compat_chat(base_url: str, api_key: str | None = None):
    """Return chat_fn for any OpenAI-compatible endpoint (Groq, Ollama, LM Studio).

    429s honor Retry-After and back off; other 4xx fail fast; 5xx retry.
    """
    import urllib.error
    import urllib.request

    ssl_ctx = _ssl_context()

    def chat(model, system, user_message, max_tokens=HF_MAX_TOKENS,
             temperature=0.7, retries=8):
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "gpt-oss" in model:
            # Reasoning model: keep hidden reasoning short so max_tokens
            # stays available for the visible answer being measured.
            body["reasoning_effort"] = "low"
        payload = json.dumps(body).encode()
        headers = dict(HTTP_HEADERS)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        for attempt in range(retries):
            req = urllib.request.Request(
                f"{base_url}/chat/completions", data=payload, headers=headers
            )
            try:
                # 180s: long enough for a 2400-token completion, short enough that
                # a connection held open by server-side throttling self-clears.
                with urllib.request.urlopen(req, timeout=180, context=ssl_ctx) as resp:
                    body = json.loads(resp.read())
                choice = body["choices"][0]
                usage = body.get("usage", {})
                return {
                    "response_text": THINK_RE.sub("", choice["message"]["content"] or ""),
                    "stop_reason": choice.get("finish_reason"),
                    "output_tokens": usage.get("completion_tokens"),
                    "input_tokens": usage.get("prompt_tokens"),
                }
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == retries - 1:
                    raise RuntimeError(f"HTTP {e.code} from {base_url}: {detail}") from e
                retry_after = e.headers.get("retry-after")
                delay = (float(retry_after) + 1 if retry_after
                         else min(5 * 2**attempt + random.uniform(0, 3), 300))
                print(f"    HTTP {e.code}, retry {attempt + 1}/{retries - 1} in {delay:.0f}s")
                time.sleep(delay)
            except urllib.error.URLError as e:
                if attempt == retries - 1:
                    raise SystemExit(
                        f"Cannot reach {base_url} ({e.reason}). "
                        "Is the server running / are you online?"
                    ) from e
                delay = min(5 * 2**attempt + random.uniform(0, 3), 120)
                print(f"    connection problem ({e.reason}), retry "
                      f"{attempt + 1}/{retries - 1} in {delay:.0f}s")
                time.sleep(delay)
            except (TimeoutError, OSError) as e:
                # Read-phase stall (server holding the connection open).
                if attempt == retries - 1:
                    raise
                delay = min(5 * 2**attempt + random.uniform(0, 3), 120)
                print(f"    request stalled ({type(e).__name__}), retry "
                      f"{attempt + 1}/{retries - 1} in {delay:.0f}s")
                time.sleep(delay)

    return chat


def make_groq_chat():
    return openai_compat_chat(GROQ_API_BASE, groq_api_key())


def groq_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set GROQ_API_KEY (free key from https://console.groq.com/keys)."
        )
    return api_key


# Fallback order when the requested model isn't on the account's catalog.
# Llama production models are Enterprise-gated on Groq; gpt-oss is open to
# developer accounts.
GROQ_SUBJECT_PREFS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
]
GROQ_JUDGE_PREFS = [
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
]


def groq_available_models() -> set[str]:
    import urllib.request

    req = urllib.request.Request(
        f"{GROQ_API_BASE}/models",
        headers={**HTTP_HEADERS, "Authorization": f"Bearer {groq_api_key()}"},
    )
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
        data = json.loads(resp.read())
    return {m["id"] for m in data.get("data", [])}


def resolve_groq_model(requested: str, prefs: list[str]) -> str:
    """Use the requested model if the account can see it, else the best fallback."""
    available = groq_available_models()
    if requested in available:
        return requested
    for candidate in prefs:
        if candidate in available:
            print(f"Model {requested} not available on this account; using {candidate}")
            return candidate
    raise SystemExit(
        f"Model {requested} not available and no known fallback found. "
        f"Models on this account: {sorted(available)}"
    )


# ------------------------------------------------------- sequential runner


def load_done(out_path: Path) -> set[str]:
    """Custom IDs of successfully collected responses (errors are retried)."""
    if not out_path.exists():
        return set()
    return {
        rec["custom_id"]
        for line in out_path.read_text().splitlines()
        if line.strip() and (rec := json.loads(line)).get("result_type") == "succeeded"
    }


def build_units(args):
    """Flat shuffled list of (custom_id, meta, system, user_message, rep)."""
    units = [
        (f"{prefix}_r{rep}", meta, system, user_message, rep)
        for prefix, meta, system, user_message in build_cells(args.suite)
        for rep in range(1, args.reps + 1)
        if not args.control_only or meta.get("skill_level") == "control"
    ]
    random.Random(args.shuffle_seed).shuffle(units)
    return units


def run_sequential(args, chat_fn):
    out_path = OUT / SUITES[args.suite]["out"]
    err_path = out_path.with_name(out_path.stem.replace("responses", "errors") + ".log")
    done = load_done(out_path)
    if done:
        print(f"Resuming: {len(done)} responses already collected")

    units = build_units(args)
    total = len(units)
    n_done, consecutive_errors = len(done), 0
    with out_path.open("a") as f:
        for custom_id, meta, system, user_message, rep in units:
            if custom_id in done:
                continue
            try:
                result = chat_fn(args.model, system, user_message,
                                 max_tokens=args.max_tokens,
                                 temperature=TEMPERATURE)
            except SystemExit:
                raise
            except Exception as e:
                consecutive_errors += 1
                msg = f"{custom_id}: {type(e).__name__}: {e}"
                with err_path.open("a") as ef:
                    ef.write(msg + "\n")
                print(f"  ERROR {msg[:160]}")
                if consecutive_errors >= 5:
                    raise SystemExit(
                        "Aborting after 5 consecutive failures - fix the cause "
                        f"and rerun to resume ({n_done}/{total} collected, "
                        f"details in {err_path})."
                    )
                continue
            consecutive_errors = 0
            record = {"custom_id": custom_id, **meta, "rep": rep,
                      "result_type": "succeeded", "model": args.model,
                      "temperature": TEMPERATURE,
                      "shuffle_seed": args.shuffle_seed, **result}
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_done += 1
            if n_done % 25 == 0 or n_done == total:
                print(f"  {n_done}/{total}")
    print(f"Done: {n_done}/{total} responses in {out_path}")


# ------------------------------------------------------ Anthropic backend


def make_anthropic_client():
    import anthropic

    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    return anthropic.Anthropic(default_headers=headers)


def build_anthropic_requests(args):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests, index = [], {}
    for custom_id, meta, system, user_message, rep in build_units(args):
        index[custom_id] = {**meta, "rep": rep}
        requests.append(
            Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=args.model,
                    max_tokens=ANTHROPIC_MAX_TOKENS,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                ),
            )
        )
    return requests, index


def run_anthropic(args):
    client = make_anthropic_client()
    requests, index = build_anthropic_requests(args)

    if args.resume:
        batch_id = args.resume
    else:
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        (OUT / "batch_id.txt").write_text(batch_id)
        print(f"Submitted batch {batch_id}")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        print(f"  status={batch.processing_status} processing={batch.request_counts.processing}")
        time.sleep(60)

    out_path = OUT / SUITES[args.suite]["out"]
    n_ok = n_err = 0
    with out_path.open("w") as f:
        for result in client.messages.batches.results(batch_id):
            meta = index.get(result.custom_id, {})
            record = {"custom_id": result.custom_id, **meta, "result_type": result.result.type}
            if result.result.type == "succeeded":
                msg = result.result.message
                record["response_text"] = "".join(
                    b.text for b in msg.content if b.type == "text"
                )
                record["stop_reason"] = msg.stop_reason
                record["output_tokens"] = msg.usage.output_tokens
                record["input_tokens"] = msg.usage.input_tokens
                record["model"] = msg.model
                n_ok += 1
            else:
                n_err += 1
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {out_path}: {n_ok} succeeded, {n_err} not succeeded")


# ---------------------------------------------------------------- entrypoint


def make_chat(backend: str):
    """Return chat_fn(model, system, user_message, **kw) for a sequential backend."""
    if backend == "groq":
        return make_groq_chat()
    if backend == "local":
        return openai_compat_chat(LOCAL_API_BASE)
    client = make_hf_client()
    return lambda model, system, user_message, **kw: hf_chat(
        client, model, system, user_message, **kw
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), default="education",
                    help="education = v1 questions; skill = v2 coding tasks")
    ap.add_argument("--backend", choices=["groq", "local", "hf", "anthropic"],
                    default="groq")
    ap.add_argument("--model", default=None, help="defaults per backend")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="completion budget (defaults per suite)")
    ap.add_argument("--shuffle-seed", type=int, default=20261014,
                    help="seed for the shuffled execution order (recorded per response)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tasks-file", default=None,
                    help="task file override for coding suites (e.g. tasks_cf_selected.json)")
    ap.add_argument("--control-only", action="store_true",
                    help="only the control cell of each task (difficulty screening)")
    ap.add_argument("--resume", metavar="BATCH_ID",
                    help="anthropic backend: re-poll an existing batch (local/hf "
                         "backends resume automatically from the responses file)")
    args = ap.parse_args()
    args.model = args.model or DEFAULT_MODELS[args.backend]
    args.max_tokens = args.max_tokens or SUITES[args.suite]["max_tokens"]
    if args.tasks_file:
        global TASK_FILE_OVERRIDE
        TASK_FILE_OVERRIDE = args.tasks_file

    OUT.mkdir(exist_ok=True)
    cells = list(build_cells(args.suite))
    if args.control_only:
        cells = [c for c in cells if c[1].get("skill_level") == "control"]
    total = len(cells) * args.reps
    print(f"Matrix [{args.suite}]: {len(cells)} cells x {args.reps} reps = "
          f"{total} requests on {args.model} via {args.backend} "
          f"(max_tokens={args.max_tokens}, temperature={TEMPERATURE}, "
          f"shuffle_seed={args.shuffle_seed})")

    if args.dry_run:
        for pick in (0, len(cells) // 2):
            prefix, meta, system, user_message = cells[pick]
            print(json.dumps({"example_cell": prefix, "meta": meta,
                              "system": system, "user_message": user_message},
                             indent=2))
        texts = {}
        key = "task_id" if args.suite in TASK_FILES else "question_id"
        for _, meta, _, user_message in cells:
            texts.setdefault(meta[key], set()).add(user_message.split("\n\n")[-1])
        bad = {k: len(v) for k, v in texts.items() if len(v) != 1}
        print("byte-identical task text across conditions:",
              "OK" if not bad else f"FAIL {bad}")
        return

    if args.backend == "anthropic":
        run_anthropic(args)
    else:
        if args.backend == "groq":
            args.model = resolve_groq_model(args.model, GROQ_SUBJECT_PREFS)
            print(f"Running on {args.model}")
        run_sequential(args, make_chat(args.backend))


if __name__ == "__main__":
    main()
