from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from env import JSBSimConfig, JSBSimRLEnv


def run(steps: int, seed: int, output: Path) -> None:
    rng = np.random.default_rng(seed)
    env = JSBSimRLEnv(JSBSimConfig(max_steps=steps))
    env.reset()
    rows = []
    total_reward = 0.0

    for _ in range(steps):
        # Small bounded perturbations around roughly level-flight controls.
        action = np.asarray(
            [
                rng.normal(0.0, 0.08),
                rng.normal(0.0, 0.08),
                rng.normal(0.0, 0.04),
                0.35 + rng.normal(0.0, 0.05),
            ],
            dtype=np.float32,
        )
        _observation, reward, done, info = env.step(action)
        total_reward += reward
        rows.append({**info, "reward": reward})
        if done:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    last = rows[-1]
    print(f"steps={len(rows)} total_reward={total_reward:.2f}")
    print(
        "final: "
        f"altitude={last['altitude_ft']:.1f} ft, "
        f"airspeed={last['airspeed_kts']:.1f} kt, "
        f"roll={last['roll_deg']:.1f} deg, "
        f"pitch={last['pitch_deg']:.1f} deg"
    )
    print(f"trajectory={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Public JSBSim RL sandbox demo")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/trajectory.csv"))
    args = parser.parse_args()
    run(args.steps, args.seed, args.output)
