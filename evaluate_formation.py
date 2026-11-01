from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from env import TwoAircraftConfig, TwoAircraftFormationEnv

Policy = Callable[[np.ndarray], np.ndarray]


def level_flight_policy(_observation: np.ndarray) -> np.ndarray:
    """Fixed public baseline control: hold the initialized level-flight state."""
    return np.asarray([0.0, 0.0, 0.0, -0.2], dtype=np.float32)


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
        "mean_reward": float(np.mean(rewards)),
        "total_reward": float(np.sum(rewards)),
        "termination": str(trajectory[-1]["termination"]),
    }
    return trajectory, summary


def run(steps: int, output_dir: Path, script: str) -> dict[str, float | int | str]:
    config = TwoAircraftConfig(max_steps=steps, script=script)
    rows, summary = run_episode(level_flight_policy, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "formation_trajectory.csv"
    summary_path = output_dir / "formation_summary.json"

    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"trajectory={trajectory_path}")
    print(f"summary={summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the public two-aircraft JSBSim baseline")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/formation"))
    parser.add_argument("--script", default="scripts/c1723.xml")
    args = parser.parse_args()
    run(args.steps, args.output_dir, args.script)
