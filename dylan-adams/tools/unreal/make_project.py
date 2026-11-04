"""Create the Unreal project for the avatar, with the plugins we need enabled.

    python tools/unreal/make_project.py            # -> unreal/VRFace/VRFace.uproject
    python tools/unreal/make_project.py --run SCRIPT.py   # run an editor Python script headless

Copies Epic's blank Blueprint template (the same thing the launcher's "New
Project" does), names it, and enables: Python editor scripting, Live Link,
Apple ARKit Face Support (receives our UDP packets on port 11111, the same
format the iPhone Live Link Face app sends), and the MetaHuman plugins
(Creator, Animator, MetaHuman Live Link).
"""

import argparse
import json
import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UE = r"C:\Program Files\Epic Games\UE_5.8"
TEMPLATE = os.path.join(UE, "Templates", "TP_BlankBP")
EDITOR_CMD = os.path.join(UE, "Engine", "Binaries", "Win64", "UnrealEditor-Cmd.exe")

PLUGINS = [
    "PythonScriptPlugin", "EditorScriptingUtilities",
    "LiveLink", "LiveLinkControlRig", "AppleARKit", "AppleARKitFaceSupport",
    "MetaHumanCharacter", "MetaHuman", "MetaHumanLiveLink",
]


def make(name, dest):
    proj_dir = os.path.join(dest, name)
    uproject = os.path.join(proj_dir, name + ".uproject")
    if os.path.exists(uproject):
        print(f"{uproject} already exists - only updating plugins")
    else:
        shutil.copytree(TEMPLATE, proj_dir, ignore=shutil.ignore_patterns("TemplateDefs.ini", "Media"))
        os.rename(os.path.join(proj_dir, "TP_BlankBP.uproject"), uproject)
    with open(uproject) as f:
        desc = json.load(f)
    have = {p["Name"] for p in desc.get("Plugins", [])}
    desc.setdefault("Plugins", []).extend({"Name": p, "Enabled": True} for p in PLUGINS if p not in have)
    for p in desc["Plugins"]:
        if p["Name"] in PLUGINS:
            p["Enabled"] = True
    with open(uproject, "w") as f:
        json.dump(desc, f, indent="\t")

    # Open on the face stage (built by setup_stage.py); Python developer mode writes type
    # stubs so editor scripts can be checked/autocompleted.
    eng = os.path.join(proj_dir, "Config", "DefaultEngine.ini")
    with open(eng) as f:
        ini = f.read()
    stage = "/Game/VRFace/FaceStage.FaceStage"
    ini = re.sub(r"(?m)^(GameDefaultMap|EditorStartupMap)=.*\n", "", ini)
    ini = ini.replace("[/Script/EngineSettings.GameMapsSettings]\n",
                      f"[/Script/EngineSettings.GameMapsSettings]\nEditorStartupMap={stage}\n"
                      f"GameDefaultMap={stage}\n")
    # The ARKit face plugin only listens for Live Link Face packets (UDP 11111, what
    # live.py --unreal sends) when this is on - see AppleARKitFaceSupportModule.cpp.
    if "bEnableLiveLinkForFaceTracking" not in ini:
        ini += ("\n[/Script/AppleARKit.AppleARKitSettings]\nbEnableLiveLinkForFaceTracking=True\n"
                "LiveLinkPublishingPort=11111\n")
    if "PythonScriptPluginSettings" not in ini:
        ini += "\n[/Script/PythonScriptPlugin.PythonScriptPluginSettings]\nbDeveloperMode=True\n"
    with open(eng, "w") as f:
        f.write(ini)
    print(f"Project ready: {uproject}")
    return uproject


def run_script(uproject, script):
    """Run an editor Python script headless (no window)."""
    cmd = [EDITOR_CMD, uproject, "-run=pythonscript", f"-script={os.path.abspath(script)}",
           "-unattended", "-nop4", "-nosplash", "-stdout", "-FullStdOutLogOutput",
           "-AllowCommandletRendering"]  # GPU texture bakes (MetaHuman assembly) need rendering
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="VRFace")
    p.add_argument("--dest", default=os.path.join(ROOT, "unreal"))
    p.add_argument("--run", help="editor Python script to run headless in the project")
    args = p.parse_args()
    uproject = make(args.name, args.dest)
    if args.run:
        raise SystemExit(run_script(uproject, args.run))


if __name__ == "__main__":
    main()
