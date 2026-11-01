from __future__ import annotations

import numpy as np


class FormationController:
    """Bounded proportional controller for the public two-aircraft task.

    The controller uses range error to adjust throttle and relative bearing to
    coordinate heading. Altitude and attitude terms damp vertical/roll motion.
    This is a transparent baseline controller, not a learned policy.
    """

    def __init__(
        self,
        range_gain: float = 2.5,
        bearing_gain: float = 1.2,
        altitude_gain: float = 0.35,
        attitude_damping: float = 0.25,
        base_throttle: float = -0.2,
    ) -> None:
        self.range_gain = range_gain
        self.bearing_gain = bearing_gain
        self.altitude_gain = altitude_gain
        self.attitude_damping = attitude_damping
        self.base_throttle = base_throttle

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        state = np.asarray(observation, dtype=np.float32).reshape(8)
        range_error, relative_bearing, altitude_error, _speed, roll, pitch, roll_rate, pitch_rate = state

        aileron = self.bearing_gain * relative_bearing - self.attitude_damping * (roll + roll_rate)
        elevator = -self.altitude_gain * altitude_error - self.attitude_damping * (pitch + pitch_rate)
        throttle = self.base_throttle + self.range_gain * range_error
        action = np.asarray([elevator, aileron, 0.0, throttle], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)
