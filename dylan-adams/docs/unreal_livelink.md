# Driving a MetaHuman face in real time from Python (UE 5.8)

Researched 2026-10-15 for UE 5.8.3 on Windows with an RTX 5070 Ti laptop. The input is a Python program that outputs 52 ARKit-style blendshapes (MediaPipe naming) plus head rotation at 30 fps.

Legend: **[V]** means verified against an official page or source code I read. **[P]** means partially verified, from community or older sources. **[U]** means unverified, my inference, or needs testing on your machine.

---

## TL;DR

- **Simplest transport: emulate the Live Link Face app in ARKit mode over UDP port 11111.** Send one datagram per frame with 61 big-endian floats. Python needs no plugin or C++ code, and UE needs only the stock **Apple ARKit Face Support** and **Live Link** plugins. The format is stable from UE 4.27 through 5.7. It still exists in 5.6+, and Epic's realtime page mentions the default "ARKit" source. **[P]** for 5.8: test it on day 1.
- **Watch out:** the new 5.6+ MetaHuman pipeline uses MetaHuman rig curves (`CTRL_expressions_*`), not ARKit curves. Epic's own sources (MetaHuman Video, Audio, and the Live Link Face "MetaHuman Animator" mode on port 14785) produce those curves. ARKit subjects still drive MetaHumans through the legacy ARKit mapping path (the "ARKit" checkbox on the MetaHuman BP's Live Link section, plus the `mh_arkit_mapping_pose` pose asset), but community reports of head-deformation and "Subject is invalid" problems in 5.7 are common. **[P]**
- **Fallback that avoids the ARKit path:** remap ARKit to `CTRL_expressions_*` yourself (a community mapping table exists) and push the curves in through a small custom Live Link source. Options: the JSONLiveLink plugin, a ~200-line C++ source, or OSC with an Anim BP **Modify Curve** node.
- **Epic already ships** real-time **webcam-to-face** and **mic-to-face** (MetaHuman Live Link plugin, 5.6+, Windows/macOS, GPU) with mood override. 5.8 adds a **new real-time audio model**, RTSP video input, and better solve models. Epic publishes no latency or quality numbers. These sources make a good **baseline** to compare against.
- **Likeness of a real person:** a MetaHuman Identity needs **iPhone/iPad TrueDepth depth footage or a stereo pair**. Mono webcam is not documented for identity, only for performance. The alternative is Mesh-to-MetaHuman from any head mesh (for example, a photogrammetry or FaceBuilder mesh).
- **Performance:** one close-up MetaHuman on an RTX 5070 Ti laptop should hold 60 fps at 1080p with the "UE Optimized" assembly, card hair, and TSR. That figure is an estimate (**[U]**); no official benchmark exists. 90 fps stereo VR needs aggressive cuts: forward renderer or no Lumen, face LOD 1 or higher, no strand hair.

---

## 1. Transport options

### 1a. Emulate Live Link Face (ARKit mode), UDP :11111 (recommended first)

**How UE receives it.** The Apple ARKit Face Support plugin's Live Link source listens on UDP 11111. Each packet becomes a Live Link Basic/Face subject whose name comes from the packet. The 61 values arrive as curves named after the ARKit enum. Epic's 5.6+ realtime page still mentions the default "ARKit" source in the Live Link window, so the path still exists **[V]** (https://dev.epicgames.com/documentation/metahuman/realtime-animation-for-metahumans-in-unreal-engine). A Jan 2026 UE 5.7 thread confirms that ARKit-mode streaming does **not** go through the new "Live Link Face" source (port 14785), which expects "MetaHuman Animator" mode. Mixing the two causes the "Subject is invalid" error **[P]** (https://forums.unrealengine.com/t/live-link-arkit-animation-issues-in-unreal-5-7-cant-get-device-recognized-subject-is-invalid/2690126).

**Packet format (packet version 6).** This is reconstructed from PyLiveLinkFace's `encode()`/`decode()` (https://github.com/JimWest/PyLiveLinkFace, file `pylivelinkface/pylivelinkface.py`, read in full) and the LLV frame description (https://github.com/flashpoint493/LLV). All multi-byte fields are **big-endian** (UE network serializer):

| Field | Type | Notes |
|---|---|---|
| PacketVersion | uint8 | `6` |
| DeviceId length | int32 BE | e.g. 36 |
| DeviceId | UTF-8 bytes | any stable UUID string |
| SubjectName length | int32 BE | |
| SubjectName | UTF-8 bytes | appears as the Live Link subject name |
| FrameTime.FrameNumber | int32 BE | timecode frame |
| FrameTime.SubFrame | float32 BE | 0.0 is fine |
| FrameRate.Numerator | int32 BE | e.g. 30 |
| FrameRate.Denominator | int32 BE | 1 |
| BlendShapeCount | uint8 | **61** |
| Values | 61 x float32 BE | order below |

A quirk in PyLiveLinkFace: it packs the version as `'<I'` (4 bytes, `06 00 00 00`), then writes `"$" + uuid` without a length prefix. This works by accident. The first 4 bytes read as `uint8 6` plus the first 3 bytes of a big-endian int32, and `'$'` = 0x24 = 36, the UUID's length. So PyLiveLinkFace writes the same bytes as the table above, provided the device id is exactly 36 characters. Use the explicit layout in your own code. The packet size range in LLV (264–774 bytes) is consistent with this layout. **[V by source reading; untested on 5.8]**

**Curve order (indices 0–60)** **[V]** (PyLiveLinkFace `FaceBlendShape` enum, which matches UE's `EARFaceBlendShape`):

```
 0 EyeBlinkLeft      1 EyeLookDownLeft   2 EyeLookInLeft     3 EyeLookOutLeft    4 EyeLookUpLeft
 5 EyeSquintLeft     6 EyeWideLeft       7 EyeBlinkRight     8 EyeLookDownRight  9 EyeLookInRight
10 EyeLookOutRight  11 EyeLookUpRight   12 EyeSquintRight   13 EyeWideRight     14 JawForward
15 JawLeft          16 JawRight         17 JawOpen          18 MouthClose       19 MouthFunnel
20 MouthPucker      21 MouthLeft        22 MouthRight       23 MouthSmileLeft   24 MouthSmileRight
25 MouthFrownLeft   26 MouthFrownRight  27 MouthDimpleLeft  28 MouthDimpleRight 29 MouthStretchLeft
30 MouthStretchRight 31 MouthRollLower  32 MouthRollUpper   33 MouthShrugLower  34 MouthShrugUpper
35 MouthPressLeft   36 MouthPressRight  37 MouthLowerDownLeft 38 MouthLowerDownRight 39 MouthUpperUpLeft
40 MouthUpperUpRight 41 BrowDownLeft    42 BrowDownRight    43 BrowInnerUp      44 BrowOuterUpLeft
45 BrowOuterUpRight 46 CheekPuff        47 CheekSquintLeft  48 CheekSquintRight 49 NoseSneerLeft
50 NoseSneerRight   51 TongueOut        52 HeadYaw          53 HeadPitch        54 HeadRoll
55 LeftEyeYaw       56 LeftEyePitch     57 LeftEyeRoll      58 RightEyeYaw      59 RightEyePitch
60 RightEyeRoll
```

MediaPipe mapping: MediaPipe names are the same words in lowerCamelCase (`jawOpen` becomes `JawOpen`). MediaPipe's 52 outputs include `_neutral` and have **no `tongueOut`** (from my knowledge of the MediaPipe FaceLandmarker; **[U]** on your version), so send 0 for index 51 and drop `_neutral`. Units: blendshapes are 0..1. PyLiveLinkFace documents head yaw/pitch/roll as **-1..1**, and the iPhone app's exact unit (normalized vs. radians) is **[U]**. Calibrate the sign and scale empirically in UE.

**Minimal sender:**

```python
import socket, struct, uuid
DEV = str(uuid.uuid4()).encode()          # 36 bytes
NAME = b"PyFace"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def send(values61, frame, fps=30, host="127.0.0.1", port=11111):
    pkt  = struct.pack("!B", 6)
    pkt += struct.pack("!i", len(DEV)) + DEV
    pkt += struct.pack("!i", len(NAME)) + NAME
    pkt += struct.pack("!ifii", frame, 0.0, fps, 1)
    pkt += struct.pack("!B61f", 61, *values61)
    sock.sendto(pkt, (host, port))
```

**Open-source implementations**
- PyLiveLinkFace (MIT, Python, tested on UE 4.27): https://github.com/JimWest/PyLiveLinkFace
- MeFaMo, MediaPipe to Live Link Face (uses pylivelinkface): https://github.com/mrassi/livelinkface, with forks https://github.com/PuddlePumpkin/MeFaMo and https://github.com/flashpoint493/MeFaMo. Epic forum post (~2022): https://forums.unrealengine.com/t/livelinkface-facial-mocap-with-any-webcam/509268
- LLV, record/replay of Live Link Face frames: https://github.com/flashpoint493/LLV
- effectustasi/metahuman-face-capture (MediaPipe over UDP :11111 to Blender; notes MediaPipe needs Python 3.11): https://github.com/effectustasi/metahuman-face-capture
- None of these claims explicit 5.6–5.8 testing. **[U]**

Firewall: allow inbound UDP 11111 for UnrealEditor.exe (https://forums.unrealengine.com/t/518053).

### 1b. Live Link Hub

- Live Link Hub is a standalone program that ships with UE. It hosts Live Link sources and rebroadcasts them to editors, and it includes Capture Manager. Custom sources come from **C++ plugins** whose `.uplugin` lists `"SupportedPrograms": ["LiveLinkHub"]`. Add the plugin folder under *User Plugin Directories* and restart **[V]** (https://dev.epicgames.com/documentation/unreal-engine/live-link-hub-ui-reference-for-unreal-engine). A source is a `ULiveLinkSourceFactory` plus an `ILiveLinkSource` **[V]** (https://dev.epicgames.com/documentation/en-us/unreal-engine/live-link-plugin-development-in-unreal-engine).
- I found **no built-in OSC/JSON input and no Python API/SDK** for Live Link Hub. **[P]**: absence of evidence.
- Epic recommends running the MetaHuman realtime sources in the **editor**, not Hub, unless Hub runs on a second machine, because of GPU contention **[V]** (realtime page above).
- Verdict: Hub adds nothing for a single-PC setup.

### 1c. Other in-engine routes

| Route | Effort | Notes |
|---|---|---|
| Live Link **Message Bus** source | High | Uses UE's UDP Messaging with UE serialization. No official Python client. Not practical from Python **[P]** (https://dev.epicgames.com/documentation/unreal-engine/live-link-in-unreal-engine) |
| **JSONLiveLink** plugin (community) | Low–Med | JSON over UDP becomes Live Link subjects (transform/camera/light/character roles). An example rather than a production tool, and may need rebuilding for 5.8 **[P]** (https://github.com/moversevlad/JSONLiveLink). The advantage is that you choose **any curve names**, e.g. `CTRL_expressions_jawOpen`. |
| **OSC plugin** (built in) | Low | Python `python-osc` sends to an OSC Server in a Blueprint, which stores a float map, then the Face Anim BP **Modify Curve** node or a Control Rig applies it. Bypasses Live Link entirely, and is the most transparent to debug. Standard UE pattern, but I found no 5.8-specific write-up **[U]**. |
| **Custom C++ Live Link source** | Med | ~200 lines (UDP socket, then `PushSubjectFrameData` with Basic role curves). Most robust, and works in editor and packaged builds. |
| VMC protocol (VMC4UE etc.) | Med | VRM-oriented. Uses ARKit "perfect sync" names over OSC. Third-party, so no advantage over the routes above. **[P]** |

**Which curve names to drive.** For a 5.6+ MetaHuman you either:
1. **Send ARKit names** (1a) and rely on the MetaHuman BP's ARKit path. The legacy Face_AnimBP feeds Live Link into the `mh_arkit_mapping_pose` pose asset (https://forums.unrealengine.com/t/metahuman-live-link-face-only-one-side-of-face-working/581352). That asset is known to lack `MouthClose` if regenerated **[P]**.
2. **Send MetaHuman rig curves** `CTRL_expressions_*` directly, through JSON/OSC/custom C++. The community ARKit-to-CTRL mapping table (with corrections) is in https://forums.unrealengine.com/t/get-new-5-6-livelink-metahuman-video-audio-working-with-arkit-52-faceshapes/2534692. For example, `EyeBlinkLeft` maps to `ctrl_expressions_eyeblinkl`, `JawOpen` to `ctrl_expressions_jawOpen`, and `MouthShrugLower` to `ctrl_expressions_jawchinraisedl`. The thread notes that `MouthClose` is best dropped or smoothed. This is the same curve space Epic's own webcam/audio solvers output, which helps an apples-to-apples comparison. **[P]**

**Recommendation:** start with 1a (half a day). If the ARKit path looks bad or breaks on 5.8, move to option 2 via OSC or a small C++ source.

---

## 2. Getting a MetaHuman into UE 5.8 and wiring Live Link (beginner steps)

Facts: since 5.6, MetaHuman Creator, Mesh to MetaHuman, and Animator are **built into the engine as plugins**. Auto-rigging and texture synthesis still run as **Epic cloud services**, so you need an Epic login (https://renderu.com/en/news/post/26778, https://digitalproduction.com/2025/06/05/metahumans-graduate-ready-for-unity-godot-and-the-fab-cash-register/). Licensing: MetaHuman is covered by the UE license, free under $1M revenue, and MetaHumans may be used in **any engine/DCC**, including Blender **[V]** (https://www.metahuman.com/license). Ready-made MetaHumans are also on Fab. In 5.8 the RigLogic/DNA libraries were open-sourced under MIT ("OpenRigLogic") **[V]** (https://dev.epicgames.com/documentation/metahuman/metahuman-5-8-release-notes-in-unreal-engine).

**Steps (5.8)** (steps 2–5 per the docs, other details **[P]**):
1. **Create the project.** Epic Launcher, UE 5.8, Games > Blank, C++ or Blueprint, Desktop, Maximum quality. Hardware docs require **DX12 / SM6** (https://dev.epicgames.com/documentation/metahuman/metahuman-hardware-requirements-in-unreal-engine).
2. **Enable plugins** (Edit > Plugins): *MetaHuman Creator*, *MetaHuman Animator*, *MetaHuman Live Link* (for Epic's webcam/audio sources), *Live Link*, *Live Link Control Rig*, *Apple ARKit*, *Apple ARKit Face Support* (for the UDP 11111 path), and optionally *OSC*. Restart.
3. **Create a character.** Content Browser: right-click > MetaHuman > **MetaHuman Character**. Open it and pick a preset or sculpt it. Like Blender's sculpt/shape-key mindset, Head/Body tools sculpt, and Skin/Hair/Eyes set the look. Request auto-rig/texture (cloud).
4. **Assemble** with **UE Optimized** (High/Medium/Low, 2K textures) for real-time, or UE Cine (8K) for hero shots (https://dev.epicgames.com/documentation/metahuman/assembly). This produces a character Blueprint (BP_<Name>).
5. **Preview Live Link in Creator.** Preview Settings > Animation > Animation Controller = *LiveLink*, then choose the subject **[V]** (https://dev.epicgames.com/documentation/metahuman/connecting-a-live-link-subject-to-your-metahuman).
6. **Put it in a level.** Drag BP_<Name> into the viewport. In Details > **Live Link**, set **Live Link Subject** to your subject and tick **Use Live Link** **[V]** (same page). For an **ARKit** subject (packets from 1a), also tick the **ARKit** checkbox (users report "2 checkboxes, AR KIT and Use Live Link" in 5.6) **[P]** (https://forums.unrealengine.com/t/how-do-i-use-2-livelink-sources-for-face-and-body-with-metahumans-ue5-6/2652915).
7. **Check data flow.** Window > Virtual Production > **Live Link**. Your Python subject (e.g. "PyFace") should show a green dot. Delete stale sources. To see the face move in the editor without pressing Play, the BP's "Update Animation in Editor" style setting may be needed **[U]**.
8. **Legacy (pre-5.6 or Fab/Bridge) MetaHumans:** in the BP, set Face component Anim Class = Face_AnimBP, then set **LLink Face Subj** = your subject name and **LLink Face Head** = true (https://docs.omniverse.nvidia.com/audio2face/latest/user-manual/livelink-ue-plugin.html; older but the same mechanism) **[P]**.
9. **Known 2026 issue:** head twisting/deformation with the ARKit Face option. It is unresolved in the thread (https://forums.unrealengine.com/t/live-linking-the-metahuman-through-iphone-animation-is-working-but-the-head-is-deforming-twisting-unnaturally-when-i-use-arkit-face-option-in-metahuman-under-live-link-option/2690594). Likely a head-rotation unit/scale issue, so try sending 0 for indices 52–60 first **[U]**.

---

## 3. What Epic already ships for real-time facial animation (positioning and baseline)

All from the **MetaHuman Live Link** plugin, UE 5.6+, created in the editor's Live Link panel (Add Source) **[V]** (https://dev.epicgames.com/documentation/metahuman/realtime-animation-for-metahumans-in-unreal-engine):

| Source | Input | Output | Notes |
|---|---|---|---|
| **MetaHuman (Video)** | Any mono UVC webcam, media source, or (5.8) **RTSP stream** | MetaHuman rig curves + head pose (optional head translation after "Capture Neutral") | GPU-heavy. Formats under 500x500 or 24 fps are filtered out. Smoothing uses a rolling average + **1€ filter** (configurable). Neutral calibration, head stabilization. Windows/macOS **[V]** (https://dev.epicgames.com/documentation/metahuman/using-a-metahuman-video-source-in-unreal-engine, https://dev.epicgames.com/documentation/metahuman/configuring-a-metahuman-video-subject) |
| **MetaHuman (Audio)** | Any mic/UVC audio device | Facial animation from speech; mood override (Neutral, Happy, Sad, Fear, Disgust, Anger, Surprise) + intensity | GPU. **5.8: "Audio-driven animation benefits from a new real-time model"** **[V]** (https://dev.epicgames.com/documentation/metahuman/using-a-metahuman-audio-source-in-unreal-engine, 5.8 release notes) |
| **Live Link Face** (iOS/Android app, "MetaHuman Animator" mode, port 14785) | Phone camera; the solve runs **on device** | Solved MetaHuman animation | iPhone 12+ (16/16 Pro recommended); Android 13+ with Snapdragon 8 Gen 1+ / Tensor G3–G5 **[V]** (https://dev.epicgames.com/documentation/metahuman/using-a-live-link-face-source, https://dev.epicgames.com/documentation/metahuman/metahuman-animator-capture-device-requirements) |
| Live Link Face **ARKit mode** (legacy) | iPhone TrueDepth / ARKit | 61 ARKit curves over UDP 11111 | The path your Python emulates |

Offline counterparts: the MetaHuman Performance asset processes mono/stereo/depth video and audio files (audio offline supports head motion, blink generation, and a mouth-only mask) (https://dev.epicgames.com/documentation/metahuman/audio-driven-animation?lang=en-US). 5.8 adds experimental **single-camera face+body capture** (offline; Markerless Motion Capture plugin, Windows only) **[V]** (5.8 notes).

**Not provided:** Epic publishes **no latency or accuracy numbers** for the realtime sources, and there is **no fused webcam+audio realtime source**. Video and Audio are separate subjects, and you would have to blend them yourself in the Anim BP. That gap is where your project fits **[V for the absence on the pages read; P overall]**. Hardware guidance is general: RTX 3070 / 8 GB VRAM, 16 cores, 32 GB RAM, DX12 (https://dev.epicgames.com/documentation/metahuman/metahuman-hardware-requirements-in-unreal-engine). Running Video + Audio + rendering on one laptop GPU will contend. Cap rendering with `t.maxfps 60` as Epic suggests **[V]**.

**Baseline plan:** record the same session (webcam + mic) into (a) MetaHuman Video, (b) MetaHuman Audio, and (c) your fused model streamed via UDP. Compare in the shared `CTRL_expressions_*` space (Take Recorder can record Live Link subjects) and measure latency with an on-screen clap/LED test. **[U]**: method suggestion.

---

## 4. MetaHuman likeness from footage or photos

- **MetaHuman Identity from footage** (Mesh to MetaHuman "From Video Footage"). Supported takes: **iPhone/iPad with TrueDepth** (promote neutral, left, right, and teeth frames) or a **stereo camera pair** (neutral + teeth). **Mono webcam is not listed for identity** **[V]** (https://dev.epicgames.com/documentation/metahuman/from-video-footage).
  1. Record an identity take (neutral face, head turned about ±30°, teeth showing) on the **same device class** used for performances (https://dev.epicgames.com/documentation/metahuman/metahuman-animator-identity-take-guidelines-in-unreal-engine).
  2. Ingest with **Capture Manager in Live Link Hub**.
  3. Content Browser: MetaHuman > **MetaHuman Identity**, then Create Components > From Footage, then promote frames and track.
  4. Run **Identity Solve** (cloud), then **Auto-Rig**, which outputs an SK_ mesh with DNA.
  5. In a MetaHuman Character asset, use the **Import > From Identity** tool (Windows only) to make an editable Creator character (https://dev.epicgames.com/documentation/metahuman/metahuman-creator-from-identity-tool-in-unreal-engine).
  6. Optionally run Fit Teeth > Prepare for Performance, used for offline performance solves (https://dev.epicgames.com/documentation/metahuman/create-a-metahuman-identity-in-unreal-engine).
- 5.8 changes: a calibration prompt for stereo without depth, automatic depth generation when solving performances, and Mesh to MetaHuman now conforms **head + body** from arbitrary-topology meshes in one workflow **[V]** (5.8 notes).
- **Photos-only / webcam-only route** **[P]**: build a head mesh elsewhere (photogrammetry such as RealityScan, KeenTools FaceBuilder for Blender, or an AI photo-to-3D tool), then **Import > From Mesh** (https://dev.epicgames.com/documentation/metahuman/from-mesh, https://dev.epicgames.com/documentation/metahuman/metahuman-creator-import-tools-in-unreal-engine). It works best with a realistic, neutral-expression mesh.

---

## 5. Performance on an RTX 5070 Ti laptop

There is no official fps figure for a single MetaHuman, so everything below is **[U]** unless cited.

- **Expectation:** one MetaHuman close-up, simple scene, 1080p. With **UE Optimized (High)**, Lumen on, and TSR, roughly 60–90 fps. With UE Cine + strand hair + Lumen + Nanite, roughly 30–50 fps. The 5070 Ti laptop is around RTX 4070 desktop class with 12 GB. Measure with `stat unit` and `stat gpu`.
- **Settings for a locked 30/60:**
  - Use the UE Optimized assembly (2K textures) (https://dev.epicgames.com/documentation/metahuman/assembly).
  - Use **hair cards** instead of strands. Grooms are the biggest cost (https://dev.epicgames.com/documentation/metahuman/controlling-metahuman-levels-of-detail-lods).
  - Use LODSync **Min LOD 1** if needed. This "disables certain animation features" (correctives), so check that lip detail survives **[V]** (same page). One forum test reports about +40% fps when correctives are off **[P]** (https://forums.unrealengine.com/t/the-new-3-0-metahuman-40-performance-gains-tutorial-inside/1777726).
  - Replace Lumen with baked or simple lights plus a skylight, use Screen Space GI or none, and turn ray tracing off.
  - Cap with `t.maxfps 30/60` to leave GPU headroom for Epic's realtime solvers if you run them as a baseline.
  - 5.8 adds FP16 RigLogic on Windows and cheaper cooked DNA init **[V]** (5.8 notes).
- **Stereo VR at 90 fps:** use the forward renderer + MSAA, Lumen off, Instanced Stereo, card hair, and face LOD 1–2 (https://developers.meta.com/vr/documentation/unreal/unreal-forward-renderer/). Anecdote: 8 MetaHumans at LOD 3–4 held 90 fps on an RTX 3090, and face LOD 0 up close caused hitches **[P]** (https://forums.unrealengine.com/t/metahuman-optimization/1208320 and related threads). Plan on face LOD 1 with forward shading, and accept some skin/material quality loss. Forward-rendering compatibility with 5.8 MetaHuman materials is **[U]**.

---

## Open items to test on day 1
1. Send the 1a packet from Python to UE 5.8.3 and confirm that a subject appears and the ARKit checkbox drives a 5.8 Creator MetaHuman.
2. Determine head-rotation units (radians vs. -1..1) and signs.
3. If the ARKit path is poor, switch to `CTRL_expressions_*` via OSC/Modify Curve or a C++ source.
4. Run Epic's MetaHuman Video and Audio sources on the same laptop as the baseline, and log GPU ms.
