"""The flybody proboscis in MuJoCo: joint commands in, proboscis extension out.

The model (flybody, Apache-2.0) is a whole fly whose proboscis joints (rostrum,
haustellum) are driven by position servos. Its free joint is removed so the body
is fixed in space; fixed that way, the model is unstable at its own 0.1 ms step
and stable at 0.01 ms, which is used here (stage-2b registration). flybody works
in centimetres; distances are returned in millimetres. Everything it produces is
**simulated**.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from neuromps.fetch import flybody
from neuromps.paths import RAW_ROOT

MODEL_FILE = RAW_ROOT / flybody.SOURCE / "assets" / "fruitfly.xml"
STEP_S = 1e-5
CONTROL_DT_S = 0.001
SETTLE_S = 0.2
TIP, HEAD = "labrum_left", "head"
JOINTS = ("rostrum", "haustellum")
# Extension limits (negative commands extend; measured before registration) and the opposing ones.
EXTEND = {"rostrum": -1.24, "haustellum": -1.59}
OPPOSE = {"rostrum": 0.183, "haustellum": 0.7}
CM_TO_MM = 10.0


def model_ready() -> bool:
    """Whether the fetched model file is present."""
    return MODEL_FILE.exists()


@lru_cache(maxsize=2)
def model(step_s: float = STEP_S):
    """The compiled model, free joint removed, at simulation step `step_s` (cached per process)."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(MODEL_FILE))
    for joint in [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]:
        spec.delete(joint)
    m = spec.compile()
    m.opt.timestep = step_s
    return m


def command(net: np.ndarray, joint: str) -> np.ndarray:
    """Servo command from net drive in [-1, 1]: towards the extension limit if positive."""
    net = np.clip(np.asarray(net, dtype=float), -1.0, 1.0)
    return np.where(net > 0, net * EXTEND[joint], -net * OPPOSE[joint])


def distance_mm(commands: dict[str, np.ndarray], step_s: float = STEP_S) -> np.ndarray:
    """Labellum-to-head distance (mm) at each control step, after a settle with commands at 0."""
    import mujoco

    m = model(step_s)
    d = mujoco.MjData(m)
    tip, head = m.body(TIP).id, m.body(HEAD).id
    actuators = {j: m.actuator(j).id for j in JOINTS}
    for _ in range(int(round(SETTLE_S / step_s))):
        mujoco.mj_step(m, d)
    substeps = int(round(CONTROL_DT_S / step_s))
    steps = len(next(iter(commands.values())))
    out = np.empty(steps)
    for k in range(steps):
        for joint, a in actuators.items():
            d.ctrl[a] = commands[joint][k] if joint in commands else 0.0
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        out[k] = CM_TO_MM * float(np.linalg.norm(d.xpos[tip] - d.xpos[head]))
    return out


@lru_cache(maxsize=4)
def rest_mm(steps: int = 1000, step_s: float = STEP_S) -> float:
    """The distance at the end of `steps` control steps with every command at 0."""
    return float(distance_mm({j: np.zeros(steps) for j in JOINTS}, step_s)[-1])


def extension_mm(commands: dict[str, np.ndarray], step_s: float = STEP_S) -> np.ndarray:
    """Proboscis extension (mm): the distance minus its value with every command at 0."""
    steps = len(next(iter(commands.values())))
    return distance_mm(commands, step_s) - rest_mm(steps, step_s)


def full_extension_mm(steps: int = 1000, step_s: float = STEP_S) -> float:
    """E_full: the extension after `steps` control steps with both joints at extension limits."""
    held = {j: np.full(steps, EXTEND[j]) for j in JOINTS}
    return float(extension_mm(held, step_s)[-1])
