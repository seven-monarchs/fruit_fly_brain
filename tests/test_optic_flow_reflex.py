"""
tests/test_optic_flow_reflex.py

Biologically grounded visual motion reflex using per-ommatidium data.

Key finding from spatial analysis:
  - Left eye (index 0) sees objects to the LEFT of fly (higher Y)
  - Right eye (index 1) sees objects to the RIGHT of fly (lower Y)
  - ommatidia_id_map column axis = within-eye internal projection,
    NOT world azimuth. The eye index is the correct L/R discriminator.

Signal model — rolling background subtraction (LPTC-inspired):
  Each step, maintain an exponentially weighted background per ommatidium:
    bg = bg * BG_ALPHA + curr * (1 - BG_ALPHA)

  Deviation from background = what changed from "normal":
    delta[eye, omm] = bg[eye, omm] - curr[eye, omm]
    (positive = darker than usual = object in view)

  This detects:
  - Sustained darkening as object approaches (cylinder enters field, stays)
  - Expanding looming (delta grows as object gets closer)
  - Frame-to-frame motion artifacts are smoothed out by BG_ALPHA

  Per-eye signal:
    signal_l = mean(max(0, delta[left_eye]  - threshold))
    signal_r = mean(max(0, delta[right_eye] - threshold))

  Reflex bias:
    asymmetry = signal_l - signal_r
    loom_bias = -GAIN * asymmetry
      -> left eye darker -> negative bias -> turn right (away from object)
      -> right eye darker -> positive bias -> turn left

  Improvements over simple mean-luminance reflex:
    1. Per-ommatidium — local contrast, not global mean
    2. Background-subtracted — adapts to ambient light levels
    3. Threshold per ommatidium — noise rejection
    4. Front looming = symmetric signal in both eyes (no spurious turn)

Run:
    wenv310\\Scripts\\python.exe tests/test_optic_flow_reflex.py
"""

import numpy as np
import flygym.preprogrammed as preprogrammed
from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

FOOD_POS = np.array([18.0, 12.0, 0.0])
csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]
CYL = {'type':'cylinder','size':[1.0,3.0],'rgba':[0.1,0.07,0.04,1],'contype':0,'conaffinity':0}

LOOM_THRESHOLD = 0.005   # per-ommatidium deviation to count as real signal
LOOM_GAIN      = 6.0
LOOM_DECAY     = 0.6
BG_ALPHA       = 0.85    # background adaptation rate (higher = slower to forget)
WALK_AMP       = 0.75
N_OMM          = 721


class OpticFlowReflex:
    """
    Per-ommatidium background-subtracted looming reflex.
    Maintains a rolling background per ommatidium per eye.
    """
    def __init__(self, n_omm=N_OMM, bg_alpha=BG_ALPHA,
                 threshold=LOOM_THRESHOLD, gain=LOOM_GAIN, decay=LOOM_DECAY):
        self.bg_alpha  = bg_alpha
        self.threshold = threshold
        self.gain      = gain
        self.decay     = decay
        self.bg        = None   # shape (2, n_omm) — initialised on first call
        self.persist   = 0.0

    def reset(self, vis):
        """Call with first obs['vision'] to initialise background."""
        self.bg = vis.mean(axis=2).copy()   # (2, 721)
        self.persist = 0.0

    def step(self, vis):
        """
        vis: shape (2, 721, 2)
        Returns: loom_bias (float), signal_l, signal_r
        """
        curr = vis.mean(axis=2)   # (2, 721)

        if self.bg is None:
            self.bg = curr.copy()
            return 0.0, 0.0, 0.0

        # Deviation from background (positive = darker than usual)
        delta = self.bg - curr

        # Per-ommatidium threshold
        delta_t = np.where(delta > self.threshold, delta - self.threshold, 0.0)

        signal_l = float(delta_t[0].mean())
        signal_r = float(delta_t[1].mean())

        # Asymmetry -> bias
        asymmetry = signal_l - signal_r
        new_bias  = -self.gain * asymmetry

        # Decay persisted bias and add new
        self.persist = self.persist * self.decay + new_bias

        # Update background (slow adaptation)
        self.bg = self.bg * self.bg_alpha + curr * (1.0 - self.bg_alpha)

        return self.persist, signal_l, signal_r


def get_vis(spawn_pos, geoms=[]):
    arena = OdorArena(odor_source=FOOD_POS[np.newaxis],
                      peak_odor_intensity=np.array([[500.0,0.0]]),
                      diffuse_func=lambda x: x**-2, marker_colors=[])
    for g in geoms:
        arena.root_element.worldbody.add('geom', **g)
    fly = Fly(spawn_pos=(*spawn_pos, 0.2), spawn_orientation=(0,0,0),
              contact_sensor_placements=csp, enable_olfaction=True,
              enable_adhesion=True, draw_adhesion=False, enable_vision=True)
    sim = HybridTurningController(fly=fly, arena=arena, timestep=1e-4)
    obs, _ = sim.reset()
    return obs['vision'].copy()


# ---------------------------------------------------------------------------
print("-- TEST 1: static signal — cylinder LEFT vs RIGHT vs absent --")
# Background = clear scene, signal = scene with cylinder
# This simulates the fly walking INTO the cylinder's visual field

spawn = (4.5, 4.0)
vis_clear     = get_vis(spawn)
vis_cyl_left  = get_vis(spawn, [{**CYL, 'pos':[6,6,3]}])
vis_cyl_right = get_vis(spawn, [{**CYL, 'pos':[6,2,3]}])
vis_cyl_front = get_vis(spawn, [{**CYL, 'pos':[6,4,3]}])

def static_bias(vis_bg, vis_curr, threshold=LOOM_THRESHOLD, gain=LOOM_GAIN):
    delta = vis_bg.mean(axis=2) - vis_curr.mean(axis=2)
    delta_t = np.where(delta > threshold, delta - threshold, 0.0)
    sl = float(delta_t[0].mean())
    sr = float(delta_t[1].mean())
    return -gain * (sl - sr), sl, sr

bias_l, sl_l, sr_l = static_bias(vis_clear, vis_cyl_left)
bias_r, sl_r, sr_r = static_bias(vis_clear, vis_cyl_right)
bias_f, sl_f, sr_f = static_bias(vis_clear, vis_cyl_front)

print(f"  LEFT  cyl: signal_l={sl_l:.5f} signal_r={sr_l:.5f} bias={bias_l:+.4f}")
print(f"  RIGHT cyl: signal_l={sl_r:.5f} signal_r={sr_r:.5f} bias={bias_r:+.4f}")
print(f"  FRONT cyl: signal_l={sl_f:.5f} signal_r={sr_f:.5f} bias={bias_f:+.4f}")

assert bias_l < 0,  f"FAIL: left cylinder should give negative bias, got {bias_l:.4f}"
assert bias_r > 0,  f"FAIL: right cylinder should give positive bias, got {bias_r:.4f}"
assert abs(bias_f) < abs(bias_l) * 0.3, \
    f"FAIL: front cylinder should be nearly symmetric, got bias={bias_f:.4f}"
print("  OK left=turn-right, right=turn-left, front=symmetric")

# ---------------------------------------------------------------------------
print("\n-- TEST 2: OpticFlowReflex class — background adapts correctly --")
reflex = OpticFlowReflex()
reflex.reset(vis_clear)

# Feed clear frames — background should stay near zero signal
for _ in range(5):
    b, sl, sr = reflex.step(vis_clear)
print(f"  After 5 clear frames: persist={reflex.persist:.5f}  bg_mean={reflex.bg.mean():.4f}")
assert abs(reflex.persist) < 0.05, \
    f"FAIL: clear scene should give near-zero persist, got {reflex.persist:.5f}"

# Now feed cylinder-on-left frame — should trigger negative bias
for _ in range(3):
    b, sl, sr = reflex.step(vis_cyl_left)
print(f"  After 3 left-cyl frames: signal_l={sl:.5f} signal_r={sr:.5f} persist={b:.4f}")
assert b < 0, f"FAIL: persist should be negative (turn right), got {b:.4f}"
print("  OK background subtraction correctly detects left cylinder")

# ---------------------------------------------------------------------------
print("\n-- TEST 3: bias decays when cylinder leaves visual field --")
# Start from clear background (same as test 2), prime with cylinder frames
reflex2 = OpticFlowReflex()
reflex2.reset(vis_clear)

# Prime with cylinder frames to build up persist
for _ in range(4):
    b, _, _ = reflex2.step(vis_cyl_left)
peak = b
print(f"  Peak persist after 4 cyl frames (from clear bg): {peak:.4f}")
assert peak < 0, f"FAIL: expected negative persist after left cyl, got {peak:.4f}"

# Switch back to clear scene — persist should decay toward zero
history = [peak]
for _ in range(8):
    b, _, _ = reflex2.step(vis_clear)
    history.append(b)
print(f"  Decay: {['%+.4f'%h for h in history]}")
assert abs(history[-1]) < abs(history[0]), \
    f"FAIL: persist should decay, start={history[0]:.4f} end={history[-1]:.4f}"
print("  OK persist decays when obstacle leaves visual field")

# ---------------------------------------------------------------------------
print("\n-- TEST 4: combined with odor steering stays in [0.1, 1.0] --")
for odor_turn in [-2.0, 0.0, 2.0]:
    for loom in [-0.8, 0.0, 0.8]:
        turn_bias = odor_turn + loom
        la = float(np.clip(WALK_AMP + turn_bias, 0.1, 1.0))
        ra = float(np.clip(WALK_AMP - turn_bias, 0.1, 1.0))
        assert 0.1 <= la <= 1.0 and 0.1 <= ra <= 1.0, \
            f"FAIL: out of range: odor={odor_turn} loom={loom} L={la} R={ra}"
print("  OK all combinations stay in [0.1, 1.0]")

# ---------------------------------------------------------------------------
print("\n-- TEST 5: signal strength proportional to cylinder proximity --")
reflex_far  = OpticFlowReflex()
reflex_near = OpticFlowReflex()
reflex_far.reset(vis_clear)
reflex_near.reset(vis_clear)

vis_far  = get_vis((2.0, 4.0), [{**CYL, 'pos':[6,6,3]}])  # 4.5mm from cyl
vis_near = get_vis((4.5, 4.0), [{**CYL, 'pos':[6,6,3]}])  # 2mm from cyl

b_far,  sl_far,  sr_far  = reflex_far.step(vis_far)
b_near, sl_near, sr_near = reflex_near.step(vis_near)
print(f"  Far  (4.5mm): signal_l={sl_far:.5f}  bias={b_far:+.4f}")
print(f"  Near (2mm):   signal_l={sl_near:.5f}  bias={b_near:+.4f}")
assert abs(b_near) > abs(b_far), \
    f"FAIL: closer cylinder should give stronger signal: near={b_near:.4f} far={b_far:.4f}"
print("  OK signal stronger when closer")

# ---------------------------------------------------------------------------
print("\n=== ALL TESTS PASSED ===")
print(f"OpticFlowReflex validated:")
print(f"  Background alpha={BG_ALPHA}, threshold={LOOM_THRESHOLD}, gain={LOOM_GAIN}, decay={LOOM_DECAY}")
print("  Left eye darker  -> negative bias -> turn right")
print("  Right eye darker -> positive bias -> turn left")
print("  Background adapts slowly -> detects sustained darkening")
print("  Per-ommatidium threshold rejects noise")
