"""Send blendshapes to Unreal as if from the iPhone Live Link Face app (ARKit mode).

UDP packet (version 6, big-endian), as read by Unreal's Apple ARKit Face
Support plugin on port 11111 - see docs/unreal_livelink.md for sources:
  u8 version | i32 len + device id | i32 len + subject name |
  i32 frame number, f32 subframe, i32 fps numerator, i32 fps denominator |
  u8 count (61) | 61 x f32 values (52 ARKit curves, head yaw/pitch/roll,
  left eye yaw/pitch/roll, right eye yaw/pitch/roll)
"""

import socket
import struct
import uuid

ARKIT_ORDER = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft",
    "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight",
    "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight", "JawForward",
    "JawLeft", "JawRight", "JawOpen", "MouthClose", "MouthFunnel",
    "MouthPucker", "MouthLeft", "MouthRight", "MouthSmileLeft", "MouthSmileRight",
    "MouthFrownLeft", "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft",
    "MouthStretchRight", "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper",
    "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft",
    "MouthUpperUpRight", "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft",
    "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft",
    "NoseSneerRight", "TongueOut",
]
HEAD = ["HeadYaw", "HeadPitch", "HeadRoll"]


class LiveLinkSender:
    def __init__(self, names, subject="VRFace", host="127.0.0.1", port=11111, fps=30):
        """names: our blendshape names (MediaPipe lowerCamelCase, e.g. jawOpen)."""
        lookup = {n[0].upper() + n[1:]: i for i, n in enumerate(names)}
        self.index = [lookup.get(n) for n in ARKIT_ORDER]  # None -> sent as 0 (e.g. TongueOut)
        self.device = str(uuid.uuid4()).encode()
        self.subject = subject.encode()
        self.addr = (host, port)
        self.fps = fps
        self.frame = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, weights, head=(0.0, 0.0, 0.0)):
        """weights: our blendshape vector; head: (yaw, pitch, roll)."""
        values = [float(weights[i]) if i is not None else 0.0 for i in self.index]
        values += [float(h) for h in head] + [0.0] * 6
        pkt = struct.pack("!B", 6)
        pkt += struct.pack("!i", len(self.device)) + self.device
        pkt += struct.pack("!i", len(self.subject)) + self.subject
        pkt += struct.pack("!ifii", self.frame, 0.0, self.fps, 1)
        pkt += struct.pack("!B61f", 61, *values)
        self.sock.sendto(pkt, self.addr)
        self.frame += 1
