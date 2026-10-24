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


class Session:
    """The leg advanced one control step at a time (for closed loops); `run` uses it too."""

    def __init__(self, names: list[str], settle_s: float = SETTLE_S):
        import mujoco

        self._mujoco = mujoco
        m = model()
        self.m, self.d = m, mujoco.MjData(m)
        self.order = [names.index(n) for n in muscle_names()]
        self.claw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CLAW)
        self.antenna = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ANTENNA)
        self.leg_joints = [
            j
            for j in range(m.njnt)
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("joint_LF")
        ]
        self.joint_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.leg_joints
        ]
        self.substeps = int(round(CONTROL_DT_S / m.opt.timestep))
        self.d.ctrl[:] = 1e-4
        for _ in range(int(round(settle_s / m.opt.timestep))):
            mujoco.mj_step(m, self.d)

    def angles(self) -> np.ndarray:
        """Current leg joint angles (rad), in `joint_names` order."""
        return self.d.qpos[self.m.jnt_qposadr[self.leg_joints]].copy()

    def step(
        self, row: np.ndarray, torques: dict[str, float] | None = None
    ) -> tuple[float, np.ndarray]:
        """Apply one row of activations for one control step; claw-antenna distance, angles.

        `torques` maps joint names to an external torque held for this step (a perturbation).
        """
        self.d.ctrl[:] = np.clip(np.asarray(row)[self.order], 1e-4, 1.0)
        self.d.qfrc_applied[:] = 0.0
        for joint, value in (torques or {}).items():
            dof = self.m.jnt_dofadr[
                self._mujoco.mj_name2id(self.m, self._mujoco.mjtObj.mjOBJ_JOINT, joint)
            ]
            self.d.qfrc_applied[dof] = value
        for _ in range(self.substeps):
            self._mujoco.mj_step(self.m, self.d)
        distance = float(np.linalg.norm(self.d.xpos[self.claw] - self.d.xpos[self.antenna]))
        return distance, self.angles()


def run(activation: np.ndarray, names: list[str], settle_s: float = SETTLE_S) -> Movement:
    """Settle passively, then apply `activation` (steps x muscles, one row per CONTROL_DT_S)."""
    session = Session(names, settle_s)
    distance, angles = [], []
    for row in activation:
        dist, ang = session.step(row)
        distance.append(dist)
        angles.append(ang)
    return Movement(
        distance_mm=np.asarray(distance),
        joint_angles=np.asarray(angles),
        joint_names=session.joint_names,
    )
