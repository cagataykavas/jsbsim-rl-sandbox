from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Protocol

import gymnasium as gym
import numpy as np
from env import JSBSimConfig, JSBSimRLEnv
from gymnasium import spaces


class CoreEnvironment(Protocol):
    def reset(self) -> np.ndarray: ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


CoreFactory = Callable[[JSBSimConfig], CoreEnvironment]


class JSBSimGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium boundary with explicit episode-ending semantics.

    The legacy core returns one ``done`` flag for backwards compatibility. This
    adapter derives Gymnasium's separate terminated/truncated values from typed
    evidence emitted by the core.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        config: JSBSimConfig | None = None,
        *,
        core_factory: CoreFactory = JSBSimRLEnv,
    ) -> None:
        super().__init__()
        self.config = config or JSBSimConfig()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self._core = core_factory(self.config)
        self._needs_reset = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("reset options are not supported")
        if seed is not None:
            self.action_space.seed(seed)
        observation = self._validated_observation(self._core.reset())
        self._needs_reset = False
        return observation, {"seed": seed}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("reset() must be called before step() or after an episode ends")
        validated_action = self._validated_action(action)
        observation, reward, done, info = self._core.step(validated_action)
        observation = self._validated_observation(observation)
        if not np.isfinite(reward):
            raise RuntimeError("core returned a non-finite reward")

        simulator_terminated = info.get("simulator_terminated")
        time_limit_reached = info.get("time_limit_reached")
        if not isinstance(simulator_terminated, bool) or not isinstance(time_limit_reached, bool):
            raise TypeError("core did not provide boolean termination evidence")
        if done != (simulator_terminated or time_limit_reached):
            raise RuntimeError("core done flag conflicts with termination evidence")

        terminated = simulator_terminated
        truncated = time_limit_reached and not terminated
        if terminated or truncated:
            self._needs_reset = True
        return observation, float(reward), terminated, truncated, dict(info)

    def close(self) -> None:
        self._core.close()
        self._needs_reset = True

    def _validated_action(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}")
        if not np.isfinite(value).all():
            raise ValueError("action must contain only finite values")
        if not self.action_space.contains(value):
            raise ValueError("action must remain within [-1, 1]")
        return value

    def _validated_observation(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != self.observation_space.shape:
            raise RuntimeError(
                f"core returned observation shape {value.shape}; "
                f"expected {self.observation_space.shape}"
            )
        if not np.isfinite(value).all():
            raise RuntimeError("core returned a non-finite observation")
        return value
