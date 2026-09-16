from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jsbsim
import numpy as np


@dataclass
class JSBSimConfig:
    script: str = "scripts/c1723.xml"
    target_altitude_ft: float = 3000.0
    max_steps: int = 1200
    dt: float = 1.0 / 60.0


class JSBSimRLEnv:
    """Small clean-room RL adapter around the public JSBSim Python API."""

    def __init__(self, config: JSBSimConfig | None = None):
        self.config = config or JSBSimConfig()
        self.fdm: jsbsim.FGFDMExec | None = None
        self.steps = 0

    def _new_exec(self) -> jsbsim.FGFDMExec:
        fdm = jsbsim.FGFDMExec(None)
        fdm.set_debug_level(0)
        fdm.set_dt(self.config.dt)
        if not fdm.load_script(self.config.script):
            raise RuntimeError(f"Could not load public JSBSim script: {self.config.script}")
        if not fdm.run_ic():
            raise RuntimeError("JSBSim initial-condition run failed")
        return fdm

    def reset(self) -> np.ndarray:
        self.fdm = self._new_exec()
        self.steps = 0
        return self._observation()

    def _get(self, name: str, default: float = 0.0) -> float:
        assert self.fdm is not None
        try:
            return float(self.fdm[name])
        except (KeyError, jsbsim.BaseError):
            return default

    def _observation(self) -> np.ndarray:
        altitude = self._get("position/h-sl-ft")
        airspeed = self._get("velocities/vc-kts")
        roll = self._get("attitude/phi-deg")
        pitch = self._get("attitude/theta-deg")
        heading = self._get("attitude/psi-true-deg")
        p = self._get("velocities/p-rad_sec")
        q = self._get("velocities/q-rad_sec")
        r = self._get("velocities/r-rad_sec")
        return np.asarray([
            altitude / 10000.0,
            airspeed / 250.0,
            roll / 180.0,
            pitch / 90.0,
            heading / 360.0,
            p,
            q,
            r,
        ], dtype=np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        assert self.fdm is not None
        action = np.asarray(action, dtype=np.float32).reshape(4)
        action = np.clip(action, -1.0, 1.0)
        elevator, aileron, rudder, throttle_signed = action
        throttle = float((throttle_signed + 1.0) * 0.5)
        self.fdm["fcs/elevator-cmd-norm"] = float(elevator)
        self.fdm["fcs/aileron-cmd-norm"] = float(aileron)
        self.fdm["fcs/rudder-cmd-norm"] = float(rudder)
        self.fdm["fcs/throttle-cmd-norm"] = throttle

    def _reward(self) -> float:
        altitude = self._get("position/h-sl-ft")
        roll = abs(self._get("attitude/phi-deg"))
        pitch = abs(self._get("attitude/theta-deg"))
        altitude_error = abs(altitude - self.config.target_altitude_ft)
        return float(
            1.0
            - min(1.0, altitude_error / 3000.0)
            - 0.35 * min(1.0, roll / 90.0)
            - 0.35 * min(1.0, pitch / 45.0)
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.fdm is None:
            raise RuntimeError("reset() must be called before step()")
        self._apply_action(action)
        running = bool(self.fdm.run())
        self.steps += 1
        done = (not running) or self.steps >= self.config.max_steps
        info = {
            "step": self.steps,
            "altitude_ft": self._get("position/h-sl-ft"),
            "airspeed_kts": self._get("velocities/vc-kts"),
            "roll_deg": self._get("attitude/phi-deg"),
            "pitch_deg": self._get("attitude/theta-deg"),
        }
        return self._observation(), self._reward(), done, info
