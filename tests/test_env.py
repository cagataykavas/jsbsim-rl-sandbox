from __future__ import annotations

import numpy as np
import pytest

from env import JSBSimConfig, JSBSimRLEnv, TwoAircraftConfig, TwoAircraftFormationEnv
from evaluate_formation import level_flight_policy, run_episode
from formation_controller import FormationController


def test_environment_reset_and_step() -> None:
    env = JSBSimRLEnv(JSBSimConfig(max_steps=2))

    observation = env.reset()
    next_observation, reward, done, info = env.step(np.zeros(4, dtype=np.float32))

    assert observation.shape == (8,)
    assert next_observation.shape == (8,)
    assert np.isfinite(observation).all()
    assert observation[0] == pytest.approx(0.3, abs=0.01)
    assert observation[1] == pytest.approx(0.4, abs=0.01)
    assert np.isfinite(next_observation).all()
    assert np.isfinite(reward)
    assert not done
    assert info["step"] == 1


def test_single_aircraft_episode_respects_step_limit() -> None:
    env = JSBSimRLEnv(JSBSimConfig(max_steps=2))
    env.reset()
    _, _, done, _ = env.step(np.zeros(4, dtype=np.float32))
    assert not done
    _, _, done, info = env.step(np.zeros(4, dtype=np.float32))
    assert done
    assert info["step"] == 2


def test_two_aircraft_reset_starts_at_target_separation() -> None:
    config = TwoAircraftConfig(max_steps=4, target_separation_ft=1200.0)
    env = TwoAircraftFormationEnv(config)
    observation = env.reset()

    assert observation.shape == (8,)
    assert np.isfinite(observation).all()
    distance, _, _, altitude_error = env._relative_state()
    assert distance == pytest.approx(config.target_separation_ft, abs=0.1)
    assert altitude_error == pytest.approx(0.0, abs=0.1)


def test_two_aircraft_instances_are_independent_and_step_reports_metrics() -> None:
    config = TwoAircraftConfig(max_steps=3)
    env = TwoAircraftFormationEnv(config)
    env.reset()
    assert env.leader is not env.follower

    observation, reward, done, info = env.step(np.asarray([0, 0, 0, -0.2], dtype=np.float32))
    assert observation.shape == (8,)
    assert np.isfinite(observation).all()
    assert np.isfinite(reward)
    assert not done
    assert info["step"] == 1
    assert info["separation_ft"] == pytest.approx(config.target_separation_ft, abs=0.1)
    assert info["termination"] == "running"


def test_two_aircraft_action_is_clipped_and_episode_terminates_at_limit() -> None:
    env = TwoAircraftFormationEnv(TwoAircraftConfig(max_steps=1))
    env.reset()
    observation, reward, done, info = env.step(np.asarray([20, -20, 5, -0.2], dtype=np.float32))

    assert observation.shape == (8,)
    assert np.isfinite(reward)
    assert done
    assert info["termination"] == "time_limit"


def test_two_aircraft_step_requires_reset() -> None:
    env = TwoAircraftFormationEnv()
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        env.step(np.zeros(4, dtype=np.float32))


def test_formation_evaluation_returns_reproducible_metric_schema() -> None:
    config = TwoAircraftConfig(max_steps=5)
    trajectory, summary = run_episode(level_flight_policy, config)

    assert len(trajectory) == 5
    assert summary["steps"] == 5
    assert summary["termination"] == "time_limit"
    assert summary["target_separation_ft"] == config.target_separation_ft
    assert summary["separation_rmse_ft"] < 100.0
    assert summary["minimum_separation_ft"] >= config.separation_limit_ft


def test_formation_controller_corrects_range_error_and_bounds_actions() -> None:
    controller = FormationController()
    too_far = np.asarray([0.5, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    too_close = np.asarray([-0.5, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    accelerate = controller(too_far)
    decelerate = controller(too_close)

    assert accelerate.shape == (4,)
    assert np.isfinite(accelerate).all()
    assert np.all(accelerate >= -1.0)
    assert np.all(accelerate <= 1.0)
    assert accelerate[3] > decelerate[3]


def test_proportional_controller_reduces_range_error_against_fixed_controls() -> None:
    config = TwoAircraftConfig(max_steps=12_000, initial_separation_ft=2400.0)
    _, fixed_summary = run_episode(level_flight_policy, config)
    _, controller_summary = run_episode(FormationController(), config)

    assert controller_summary["separation_rmse_ft"] < fixed_summary["separation_rmse_ft"] * 0.9
