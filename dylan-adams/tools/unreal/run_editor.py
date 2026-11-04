"""Open the VRFace project in the full Unreal editor and run a Python script in it.

    python tools/unreal/run_editor.py tools/unreal/assemble_gui.py

Unlike make_project.py --run (headless), this opens the editor window, so
scripts that need the editor to keep ticking (texture bakes, assembly) work.
The script is responsible for quitting the editor when it's done.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDITOR = r"C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor.exe"
UPROJECT = os.path.join(ROOT, "unreal", "VRFace", "VRFace.uproject")

if __name__ == "__main__":
    script = os.path.abspath(sys.argv[1])
    log = os.path.join(ROOT, "out", "logs", "unreal_editor.log")
    # -ExecCmds keeps the editor running after the script returns (-ExecutePythonScript quits),
    # so tick callbacks registered by the script get to run.
    assert " " not in script, "script path must not contain spaces (ExecCmds quoting)"
    cmd = [EDITOR, UPROJECT, f"-ExecCmds=py {script}", "-nosplash", f"-abslog={log}"]
    print(" ".join(cmd))
    raise SystemExit(subprocess.run(cmd).returncode)
