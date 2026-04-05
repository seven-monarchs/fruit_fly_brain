"""
tests/test_back_camera.py

Camera test: fly walks toward a wall, turns, walks alongside it.
No Brian2, no flyvis, no odor steering - pure hardcoded control.
10 seconds of physics, three-panel video.

Layout:
  Row 1: isometric reference (camera_top_right)
  Row 2: top-down view       (camera_top_zoomout)
  Row 3: back cam            (custom close lower-back)

Run:
    wenv310\\Scripts\\python.exe tests/test_back_camera.py
"""

import numpy as np
import imageio
from pathlib import Path
from dotenv import load_dotenv

from flygym import Fly, YawOnlyCamera
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
load_dotenv()

PHYSICS_TIMESTEP = 1e-4
PHYSICS_STEPS    = 250        # microsteps per decision (25 ms)
N_STEPS          = 400        # 400 x 25 ms = 10 s
FPS              = 30
PLAY_SPEED       = 0.25       # 4x slow-motion so legs and wall contact are clear

# Back-cam parameters — tune here if needed
BACK_CAM_POS   = [-4, 0, 3.5]
BACK_CAM_EULER = [1.1, 0.0, -1.57]

# Wall: x=6mm, spans y=-12..12, base z=-1mm (truly blocks fly body)
WALL_X = 6.0

WALK_AMP        = 0.8
TURN_STEPS      = 80    # how many steps to spend turning (80 x 25ms = 2s)
TURN_DONE_DEG   = 80    # fallback: also stop turning once heading reaches this

arena = OdorArena(
    odor_source=np.array([[20.0, 0.0, 0.0]]),
    peak_odor_intensity=np.array([[1.0, 0.0]]),
    diffuse_func=lambda x: x**-2,
    marker_colors=[],
)

# Solid wall in front of the fly (fly faces +x, heading=0)
arena.root_element.worldbody.add(
    "geom", name="side_wall",
    type="box", pos=[WALL_X, 0.0, 3.0], size=[0.3, 12.0, 4.0],
    rgba=[0.20, 0.13, 0.07, 1.0], contype=1, conaffinity=1,
    solimp="0.9 0.999 0.001 0.5 2", solref="0.02 1",
)

contact_sensor_placements = [
    f"{leg}{seg}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

fly = Fly(
    spawn_pos=(0, 0, 0.2),
    spawn_orientation=(0, 0, 0),   # facing east (+x)
    contact_sensor_placements=contact_sensor_placements,
    enable_olfaction=False,
    enable_adhesion=True,
    draw_adhesion=False,
    enable_vision=False,
)

cam_iso = YawOnlyCamera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_top_right",
    targeted_fly_names=fly.name,
    timestamp_text=False, play_speed_text=False,
    play_speed=PLAY_SPEED, fps=FPS,
)
cam_top = YawOnlyCamera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_top_zoomout",
    targeted_fly_names=fly.name,
    timestamp_text=False, play_speed_text=False,
    play_speed=PLAY_SPEED, fps=FPS,
)
cam_back = YawOnlyCamera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_back_close",
    targeted_fly_names=fly.name,
    timestamp_text=False, play_speed_text=False,
    play_speed=PLAY_SPEED, fps=FPS,
    camera_parameters={
        "class": "nmf",
        "mode":  "track",
        "ipd":   0.068,
        "pos":   BACK_CAM_POS,
        "euler": BACK_CAM_EULER,
    },
)

sim = HybridTurningController(
    fly=fly, cameras=[cam_iso, cam_top, cam_back], arena=arena,
    timestep=PHYSICS_TIMESTEP,
)
obs, _ = sim.reset()

# Get root joint to read position
all_qpos   = list(sim.physics.named.data.qpos.axes[0].names)
root_joint = next(
    (n for n in all_qpos if "dust" not in n
     and hasattr(sim.physics.named.data.qpos[n], "__len__")
     and len(sim.physics.named.data.qpos[n]) == 7),
    None,
)

print(f"Running {N_STEPS} steps ({N_STEPS * PHYSICS_STEPS * PHYSICS_TIMESTEP:.0f}s) ...")
print(f"  Steps 0-10:             walk straight")
print(f"  Steps 10-{10+TURN_STEPS}: hard left turn")
print(f"  Steps {10+TURN_STEPS}+:        cruise straight (new heading)")

for step in range(N_STEPS):
    # Read heading for logging
    fly_heading_deg = 0.0
    if root_joint:
        q = sim.physics.named.data.qpos[root_joint]
        qw, qz = float(q[3]), float(q[6])
        fly_heading_deg = float(np.degrees(np.arctan2(2*qw*qz, 1 - 2*qz**2)))

    # Control — simple timed phases, no position sensing needed
    if step < 10:
        ctrl = np.array([WALK_AMP, WALK_AMP])   # brief straight walk
    elif step < 10 + TURN_STEPS and fly_heading_deg < TURN_DONE_DEG:
        ctrl = np.array([0.0, 1.0])             # maximum left turn (stop left legs entirely)
    else:
        ctrl = np.array([WALK_AMP, WALK_AMP])   # cruise in new direction

    for _ in range(PHYSICS_STEPS):
        obs, _, _, _, _ = sim.step(ctrl)
    sim.render()

    if step % 20 == 0:
        print(f"  step {step:03d}  heading={fly_heading_deg:.0f}deg")

frames_iso  = cam_iso._frames
frames_top  = cam_top._frames
frames_back = cam_back._frames
n_frames    = len(frames_iso)
print(f"  captured {n_frames} frames")

try:
    from PIL import Image as PILImage
    def _resize(frame, w, h):
        return np.array(PILImage.fromarray(frame).resize((w, h), PILImage.LANCZOS))
except ImportError:
    import cv2
    def _resize(frame, w, h):
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

panel_w = frames_iso[0].shape[1]

def _panel_h(frames, w):
    h0, w0 = frames[0].shape[:2]
    h = int(h0 * w / w0)
    return h + h % 2

iso_h  = _panel_h(frames_iso,  panel_w)
top_h  = _panel_h(frames_top,  panel_w)
back_h = _panel_h(frames_back, panel_w)

out_path = Path(__file__).parent / "back_cam_test.mp4"
writer = imageio.get_writer(
    str(out_path), fps=FPS, codec="libx264",
    macro_block_size=None,
    output_params=["-pix_fmt", "yuv420p", "-crf", "18"],
)
for i in range(n_frames):
    r1 = _resize(frames_iso[i],  panel_w, iso_h)
    r2 = _resize(frames_top[i],  panel_w, top_h)
    r3 = _resize(frames_back[i], panel_w, back_h)
    writer.append_data(np.concatenate([r1, r2, r3], axis=0))
writer.close()

print(f"\nDone. {out_path}")
print(f"  Back cam: pos={BACK_CAM_POS}  euler={BACK_CAM_EULER}")
