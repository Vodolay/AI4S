"""The FlyMimic musculoskeletal leg in MuJoCo: muscle activation in, leg movement out.

The model (FlyGym 2, Apache-2.0) is a whole fly with an anchored thorax whose
left front leg is driven by 15 Hill-type muscles. It is prepared once from the
fetched files into `data/interim/flymimic/` (the meshes where the model file
expects them) and loaded with plain MuJoCo, so the project's Python 3.11
environment runs it without FlyGym. Everything it produces is **simulated**.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from neuromps.fetch import flymimic
from neuromps.paths import INTERIM_ROOT, RAW_ROOT

RAW = RAW_ROOT / flymimic.SOURCE
WORK = INTERIM_ROOT / "flymimic"
CLAW, ANTENNA = "LFTarsus5", "LPedicel"
CONTROL_DT_S = 0.001
SETTLE_S = 0.2
# Passive rest after the settle, measured before registration (see the stage-1 registration).
REST_DISTANCE_MM = 1.528


def model_ready() -> bool:
    """Whether the fetched model files are present."""
    return (RAW / flymimic.MODEL_NAME).exists() and (RAW / f"{flymimic.MESH_ARCHIVE}.tar").exists()


def prepare() -> Path:
    """Unpack the meshes where the model file looks for them; return the model path."""
    target = WORK / flymimic.MODEL_NAME
    if not target.exists():
        flymimic.extract_meshes(RAW / f"{flymimic.MESH_ARCHIVE}.tar", WORK / "meshes" / "stl")
        shutil.copy(RAW / flymimic.MODEL_NAME, target)
    return target


@lru_cache(maxsize=1)
def model():
    """The compiled MuJoCo model (cached per process)."""
    import mujoco

    return mujoco.MjModel.from_xml_path(str(prepare()))


def muscle_names() -> list[str]:
    """Actuator (muscle) names in actuator order."""
    import mujoco

    m = model()
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]


@dataclass(frozen=True)
class Movement:
    """Claw-antenna distance (mm) and joint angles (rad) at each control step after the settle."""

    distance_mm: np.ndarray
    joint_angles: np.ndarray
    joint_names: list[str]


def run(activation: np.ndarray, names: list[str], settle_s: float = SETTLE_S) -> Movement:
    """Settle passively, then apply `activation` (steps x muscles, one row per CONTROL_DT_S)."""
    import mujoco

    m = model()
    d = mujoco.MjData(m)
    order = [names.index(n) for n in muscle_names()]
    claw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CLAW)
    antenna = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ANTENNA)
    leg_joints = [
        j
        for j in range(m.njnt)
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("joint_LF")
    ]
    substeps = int(round(CONTROL_DT_S / m.opt.timestep))
    d.ctrl[:] = 1e-4
    for _ in range(int(round(settle_s / m.opt.timestep))):
        mujoco.mj_step(m, d)
    distance, angles = [], []
    for row in activation:
        d.ctrl[:] = np.clip(row[order], 1e-4, 1.0)
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        distance.append(float(np.linalg.norm(d.xpos[claw] - d.xpos[antenna])))
        angles.append(d.qpos[m.jnt_qposadr[leg_joints]].copy())
    return Movement(
        distance_mm=np.asarray(distance),
        joint_angles=np.asarray(angles),
        joint_names=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in leg_joints],
    )
