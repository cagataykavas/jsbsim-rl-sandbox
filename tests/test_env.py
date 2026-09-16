from __future__ import annotations

import numpy as np

from env import JSBSimConfig, JSBSimRLEnv


def test_environment_reset_and_step() -> None:
    env = JSBSimRLEnv(JSBSimConfig(max_steps=2))

    observation = env.reset()
    next_observation, reward, done, info = env.step(np.zeros(4, dtype=np.float32))

    assert observation.shape == (8,)
    assert next_observation.shape == (8,)
    assert np.isfinite(observation).all()
    assert np.isfinite(next_observation).all()
    assert np.isfinite(reward)
    assert not done
    assert info["step"] == 1
