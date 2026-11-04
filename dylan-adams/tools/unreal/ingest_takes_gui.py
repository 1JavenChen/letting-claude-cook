"""Runs INSIDE the Unreal editor (with its window): run ingest_takes.py, then quit.

    python tools/unreal/run_editor.py tools/unreal/ingest_takes_gui.py

Headless Unreal has audio disabled, and ingest fails at the very end when it
creates the take's sound asset; the editor proper has audio.
"""

import os

import unreal

state = {"frame": 0, "handle": None}
HERE = os.path.dirname(os.path.abspath(__file__))


def tick(dt):
    state["frame"] += 1
    if state["frame"] == 120:  # let the editor finish starting up
        unreal.unregister_slate_post_tick_callback(state["handle"])
        try:
            with open(os.path.join(HERE, "ingest_takes.py")) as f:
                exec(compile(f.read(), "ingest_takes.py", "exec"), {"__name__": "__main__"})
        finally:
            unreal.SystemLibrary.quit_editor()


state["handle"] = unreal.register_slate_post_tick_callback(tick)
unreal.log("[vrface] ingest_takes_gui registered")
