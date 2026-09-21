from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from env import JSBSimConfig
from gym_env import JSBSimGymEnv


class FakeCore:
    def __init__(
        self,
        config: JSBSimConfig,
        *,
        terminate_at: int | None = None,
        invalid_evidence: bool = False,
    ) -> None:
        self.config = config
        self.terminate_at = terminate_at
        self.invalid_evidence = invalid_evidence
        self.steps = 0
        self.closed = False

    def reset(self) -> np.ndarray:
        self.steps = 0
        return np.zeros(8, dtype=np.float32)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self.steps += 1
        terminated = self.terminate_at == self.steps
        truncated = self.steps >= self.config.max_steps
        info: dict[str, Any] = {
            "step": self.steps,
            "simulator_terminated": terminated,
            "time_limit_reached": truncated,
        }
        if self.invalid_evidence:
            info.pop("simulator_terminated")
        return (
            np.full(8, self.steps, dtype=np.float32),
            1.0,
            terminated or truncated,
            info,
        )

    def close(self) -> None:
        self.closed = True


def make_env(
    *,
    max_steps: int = 2,
    terminate_at: int | None = None,
    invalid_evidence: bool = False,
) -> tuple[JSBSimGymEnv, FakeCore]:
    core: FakeCore | None = None

    def factory(config: JSBSimConfig) -> FakeCore:
        nonlocal core
        core = FakeCore(
            config,
            terminate_at=terminate_at,
            invalid_evidence=invalid_evidence,
        )
        return core

    env = JSBSimGymEnv(JSBSimConfig(max_steps=max_steps), core_factory=factory)
    assert core is not None
    return env, core


def test_spaces_and_reset_follow_gymnasium_contract() -> None:
    env, _core = make_env()

    observation, info = env.reset(seed=7)

    assert isinstance(env, gym.Env)
    assert env.action_space.shape == (4,)
    assert env.observation_space.contains(observation)
    assert info == {"seed": 7}


def test_time_limit_is_truncation_not_termination() -> None:
    env, _core = make_env(max_steps=1)
    env.reset()

    _observation, _reward, terminated, truncated, info = env.step(np.zeros(4))

    assert terminated is False
    assert truncated is True
    assert info["time_limit_reached"] is True


def test_simulator_stop_takes_precedence_over_same_step_time_limit() -> None:
    env, _core = make_env(max_steps=1, terminate_at=1)
    env.reset()

    _observation, _reward, terminated, truncated, _info = env.step(np.zeros(4))

    assert terminated is True
    assert truncated is False


def test_step_after_episode_end_requires_reset() -> None:
    env, _core = make_env(max_steps=1)
    env.reset()
    env.step(np.zeros(4))

    with pytest.raises(RuntimeError, match="after an episode ends"):
        env.step(np.zeros(4))


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (np.zeros(3), "shape"),
        (np.asarray([0.0, 0.0, 0.0, np.nan]), "finite"),
        (np.asarray([0.0, 0.0, 0.0, 1.1]), "within"),
    ],
)
def test_invalid_actions_fail_closed(action: np.ndarray, message: str) -> None:
    env, _core = make_env()
    env.reset()

    with pytest.raises(ValueError, match=message):
        env.step(action)


def test_action_space_sampling_is_repeatable_after_seeded_reset() -> None:
    env, _core = make_env()

    env.reset(seed=41)
    first = env.action_space.sample()
    env.reset(seed=41)
    second = env.action_space.sample()

    np.testing.assert_array_equal(first, second)


def test_missing_termination_evidence_is_rejected() -> None:
    env, _core = make_env(invalid_evidence=True)
    env.reset()

    with pytest.raises(TypeError, match="termination evidence"):
        env.step(np.zeros(4))


def test_close_releases_core_and_requires_new_reset() -> None:
    env, core = make_env()
    env.reset()

    env.close()

    assert core.closed is True
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4))
