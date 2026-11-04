"""Runs INSIDE Unreal. Dumps enums/structs needed to script MetaHuman creation."""

import os

import unreal

out = []
for name in ["MetaHumanRigType", "MetaHumanCharacterAutoRiggingRequestParams",
             "MetaHumanCharacterEditorBuildParameters", "MetaHumanCharacterTextureRequestParams",
             "MetaHumanQualityLevel", "MetaHumanDefaultPipelineType", "MetaHumanCharacterRigState",
             "MetaHumanCharacterSkinSettings", "MetaHumanCharacterTemplateType"]:
    t = getattr(unreal, name, None)
    out.append(f"\n===== {name} =====")
    if t is None:
        out.append("  (missing)")
        continue
    out.append((t.__doc__ or "").strip()[:2500])
    if issubclass(t, unreal.EnumBase):
        out.append("  values: " + ", ".join(str(v) for v in t.__members__) if hasattr(t, "__members__") else "")
for name in sorted(dir(unreal)):
    if name.startswith("MetaHuman") and ("Enum" in type(getattr(unreal, name)).__name__ or "Rig" in name
                                         or "Pipeline" in name or "Quality" in name or "Build" in name):
        out.append(f"type: {name}")
c = unreal.MetaHumanCharacter
out.append("\n===== MetaHumanCharacter props =====\n" + (c.__doc__ or "")[:6000])
path = os.path.join(unreal.Paths.project_saved_dir(), "vrface_api2.txt")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
unreal.log(f"[vrface] api2 -> {path}")
