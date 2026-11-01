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


@dataclass
class TwoAircraftConfig:
    """Configuration for a public, non-tactical leader/follower benchmark."""

    target_altitude_ft: float = 3000.0
    target_separation_ft: float = 1500.0
    max_steps: int = 1200
    dt: float = 1.0 / 60.0
    script: str = "scripts/c1723.xml"
    separation_limit_ft: float = 500.0


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


class TwoAircraftFormationEnv:
    """Two independent public JSBSim aircraft in a simple following task.

    The leader flies a fixed, level-flight control input. The follower receives
    normalized elevator, aileron, rudder and throttle commands. Positions are
    represented in a local north/east plane derived from latitude and longitude.
    """

    def __init__(self, config: TwoAircraftConfig | None = None):
        self.config = config or TwoAircraftConfig()
        self.leader: jsbsim.FGFDMExec | None = None
        self.follower: jsbsim.FGFDMExec | None = None
        self.steps = 0
        self.initial_latitude_rad = 0.0
        self.initial_longitude_rad = 0.0
        self.initial_altitude_ft = 0.0

    def _new_exec(self, latitude_deg: float, longitude_deg: float) -> jsbsim.FGFDMExec:
        fdm = jsbsim.FGFDMExec(None)
        fdm.set_debug_level(0)
        fdm.set_dt(self.config.dt)
        if not fdm.load_script(self.config.script):
            raise RuntimeError(f"Could not load public JSBSim script: {self.config.script}")
        fdm["ic/lat-geod-deg"] = latitude_deg
        fdm["ic/long-gc-deg"] = longitude_deg
        fdm["ic/h-sl-ft"] = self.config.target_altitude_ft
        fdm["ic/vc-kts"] = 100.0
        fdm["ic/psi-true-deg"] = 90.0
        if not fdm.run_ic():
            raise RuntimeError("JSBSim initial-condition run failed")
        return fdm

    @staticmethod
    def _get(fdm: jsbsim.FGFDMExec, name: str, default: float = 0.0) -> float:
        try:
            return float(fdm[name])
        except (KeyError, jsbsim.BaseError):
            return default

    def reset(self) -> np.ndarray:
        # The leader heads north; initialize the follower behind at the target gap.
        latitude_deg = 37.0
        longitude_deg = -122.0
        longitude_offset_deg = self.config.target_separation_ft / (
            20_925_524.9 * np.cos(np.deg2rad(latitude_deg))
        ) * 180.0 / np.pi
        self.leader = self._new_exec(latitude_deg, longitude_deg)
        self.follower = self._new_exec(latitude_deg, longitude_deg - longitude_offset_deg)
        self.steps = 0
        self.initial_latitude_rad = np.deg2rad(self._get(self.leader, "position/lat-geod-deg"))
        self.initial_longitude_rad = np.deg2rad(self._get(self.leader, "position/long-gc-deg"))
        self.initial_altitude_ft = self._get(self.leader, "position/h-sl-ft")
        self._set_controls(self.leader, np.asarray([0.0, 0.0, 0.0, -0.2], dtype=np.float32))
        return self._observation()

    def _local_position(self, fdm: jsbsim.FGFDMExec) -> tuple[float, float, float]:
        lat = np.deg2rad(self._get(fdm, "position/lat-geod-deg"))
        lon = np.deg2rad(self._get(fdm, "position/long-gc-deg"))
        north_ft = (lat - self.initial_latitude_rad) * 20_925_524.9
        east_ft = (lon - self.initial_longitude_rad) * 20_925_524.9 * np.cos(self.initial_latitude_rad)
        altitude = self._get(fdm, "position/h-sl-ft")
        return float(north_ft), float(east_ft), altitude

    def _relative_state(self) -> tuple[float, float, float, float]:
        assert self.leader is not None and self.follower is not None
        leader_n, leader_e, leader_alt = self._local_position(self.leader)
        follower_n, follower_e, follower_alt = self._local_position(self.follower)
        delta_n = leader_n - follower_n
        delta_e = leader_e - follower_e
        horizontal = float(np.hypot(delta_n, delta_e))
        distance_3d = float(np.hypot(horizontal, leader_alt - follower_alt))
        bearing_error_deg = float(np.rad2deg(np.arctan2(delta_e, delta_n)))
        altitude_error = float(follower_alt - leader_alt)
        return distance_3d, horizontal, bearing_error_deg, altitude_error

    def _observation(self) -> np.ndarray:
        assert self.leader is not None and self.follower is not None
        distance, _horizontal, bearing, altitude_error = self._relative_state()
        return np.asarray([
            (distance - self.config.target_separation_ft) / self.config.target_separation_ft,
            bearing / 180.0,
            altitude_error / 1000.0,
            self._get(self.follower, "velocities/vc-kts") / 250.0,
            self._get(self.follower, "attitude/phi-deg") / 180.0,
            self._get(self.follower, "attitude/theta-deg") / 90.0,
            self._get(self.follower, "velocities/p-rad_sec"),
            self._get(self.follower, "velocities/q-rad_sec"),
        ], dtype=np.float32)

    @staticmethod
    def _set_controls(fdm: jsbsim.FGFDMExec, action: np.ndarray) -> None:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(4), -1.0, 1.0)
        elevator, aileron, rudder, throttle_signed = action
        fdm["fcs/elevator-cmd-norm"] = float(elevator)
        fdm["fcs/aileron-cmd-norm"] = float(aileron)
        fdm["fcs/rudder-cmd-norm"] = float(rudder)
        fdm["fcs/throttle-cmd-norm"] = float((throttle_signed + 1.0) * 0.5)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.leader is None or self.follower is None:
            raise RuntimeError("reset() must be called before step()")
        self._set_controls(self.follower, action)
        leader_running = bool(self.leader.run())
        follower_running = bool(self.follower.run())
        self.steps += 1
        distance, horizontal, bearing, altitude_error = self._relative_state()
        distance_error = distance - self.config.target_separation_ft
        reward = float(1.0 - min(2.0, abs(distance_error) / self.config.target_separation_ft)
                       - min(1.0, abs(altitude_error) / 1000.0))
        separation_violation = distance < self.config.separation_limit_ft
        done = (not leader_running or not follower_running or separation_violation
                or self.steps >= self.config.max_steps)
        if separation_violation:
            termination = "separation_violation"
        elif not leader_running or not follower_running:
            termination = "simulator_stop"
        elif self.steps >= self.config.max_steps:
            termination = "time_limit"
        else:
            termination = "running"
        info = {
            "step": self.steps,
            "separation_ft": distance,
            "horizontal_separation_ft": horizontal,
            "distance_error_ft": distance_error,
            "bearing_error_deg": bearing,
            "altitude_error_ft": altitude_error,
            "termination": termination,
        }
        return self._observation(), reward, done, info
