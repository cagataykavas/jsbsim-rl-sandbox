from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from env import TwoAircraftConfig, TwoAircraftFormationEnv
from formation_controller import FormationController

Policy = Callable[[np.ndarray], np.ndarray]


def level_flight_policy(_observation: np.ndarray) -> np.ndarray:
    """Fixed public baseline control: hold the initialized level-flight state."""
    return np.asarray([0.0, 0.0, 0.0, -0.2], dtype=np.float32)


def evaluate_policy(
    name: str,
    policy: Policy,
    config: TwoAircraftConfig,
    output_dir: Path,
) -> dict[str, float | int | str]:
    rows, summary = run_episode(policy, config)
    summary["policy"] = name
    trajectory_path = output_dir / f"{name}_trajectory.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def run_episode(policy: Policy, config: TwoAircraftConfig) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    env = TwoAircraftFormationEnv(config)
    observation = env.reset()
    trajectory: list[dict[str, float | int | str]] = []
    rewards: list[float] = []

    while True:
        action = policy(observation)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        trajectory.append({**info, "reward": reward})
        if done:
            break

    separation = np.asarray([float(row["separation_ft"]) for row in trajectory])
    target_error = separation - config.target_separation_ft
    summary: dict[str, float | int | str] = {
        "steps": len(trajectory),
        "target_separation_ft": config.target_separation_ft,
        "mean_separation_ft": float(np.mean(separation)),
        "separation_rmse_ft": float(np.sqrt(np.mean(target_error**2))),
        "minimum_separation_ft": float(np.min(separation)),
        "final_separation_ft": float(separation[-1]),
        "mean_reward": float(np.mean(rewards)),
        "total_reward": float(np.sum(rewards)),
        "termination": str(trajectory[-1]["termination"]),
    }
    return trajectory, summary


def run(steps: int, output_dir: Path, script: str) -> dict[str, float | int | str]:
    # Start outside the target gap to make the correction task measurable.
    config = TwoAircraftConfig(
        max_steps=steps,
        script=script,
        initial_separation_ft=2400.0,
        target_separation_ft=1500.0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "formation_summary.json"
    summaries = [
        evaluate_policy("fixed_baseline", level_flight_policy, config, output_dir),
        evaluate_policy("proportional_controller", FormationController(), config, output_dir),
    ]
    baseline_rmse = float(summaries[0]["separation_rmse_ft"])
    controlled_rmse = float(summaries[1]["separation_rmse_ft"])
    improvement = (baseline_rmse - controlled_rmse) / baseline_rmse * 100.0
    result: dict[str, float | int | str] = {
        "initial_separation_ft": config.initial_separation_ft or config.target_separation_ft,
        "target_separation_ft": config.target_separation_ft,
        "baseline_separation_rmse_ft": baseline_rmse,
        "controller_separation_rmse_ft": controlled_rmse,
        "rmse_reduction_pct": improvement,
        "controller_final_separation_ft": float(summaries[1]["final_separation_ft"]),
        "controller_minimum_separation_ft": float(summaries[1]["minimum_separation_ft"]),
        "baseline_termination": str(summaries[0]["termination"]),
        "controller_termination": str(summaries[1]["termination"]),
        "steps_per_episode": steps,
    }
    summary_path.write_text(json.dumps({"comparison": result, "runs": summaries}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": result, "runs": summaries}, indent=2))
    print(f"trajectories={output_dir}")
    print(f"summary={summary_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the public two-aircraft JSBSim baseline")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/formation"))
    parser.add_argument("--script", default="scripts/c1723.xml")
    args = parser.parse_args()
    run(args.steps, args.output_dir, args.script)
