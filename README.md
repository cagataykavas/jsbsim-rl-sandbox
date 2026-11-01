# JSBSim RL Sandbox

A clean-room, public demonstration of reinforcement-learning environment design around the open-source **JSBSim** flight-dynamics library.

This repository is intentionally independent from any employer code, data, scenarios, reward functions, model configurations or naming conventions. It uses only the public JSBSim Python API and aircraft/scripts distributed with the public package.

## Why this exists

The goal is to demonstrate the engineering boundary between a flight-dynamics simulator and an RL agent without publishing proprietary work:

```text
public JSBSim model
      |
      v
simulation adapter
      |
      +--> normalized observation
      +<-- normalized control action
      |
      v
reward / termination
      |
      v
Gym-style environment
      |
      +--> random policy demo
      +--> PPO/SAC integration point
```

## Public-origin boundary

The implementation is written specifically for this repository from public documentation. It does **not** reproduce company scenarios, internal aircraft models, tactical logic, target logic, reward values, telemetry formats, project naming or proprietary configuration.

The default example uses the public `c1723.xml` script shipped with JSBSim. You can substitute another model/script available in your own public JSBSim installation.

## Features

- Python wrapper around `jsbsim.FGFDMExec`
- public JSBSim aircraft/script loading
- normalized continuous elevator/aileron/rudder/throttle actions
- compact observation vector
- configurable episode duration
- simple altitude/attitude stabilization reward
- reset/step API suitable for RL experiments
- deterministic random-policy smoke demo
- CSV trajectory export
- two-aircraft leader/follower episode using two independent public JSBSim instances
- bounded proportional controller and fixed-control comparison with tracking-error metrics

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python demo.py --steps 800 --seed 42
python evaluate_formation.py --steps 12000 --output-dir artifacts/formation
```

The single-aircraft demo executes a bounded random policy and writes a trajectory CSV. The formation evaluation compares fixed follower controls against a bounded proportional controller and writes both trajectories plus `formation_summary.json` under the selected output directory.

## Two-aircraft benchmark

`TwoAircraftFormationEnv` runs two separately initialized `c1723.xml` JSBSim instances. The leader follows the public script and the follower accepts normalized elevator, aileron, rudder and throttle actions. Both start at 3,000 ft and 100 kt. The default evaluation deliberately starts the follower 2,400 ft behind the leader and targets a 1,500 ft gap so range correction can be measured. The scenario is a generic public-simulator following task; it does not model a tactical intercept or any employer scenario.

The evaluation reports separation RMSE for both policies and the percentage reduction against fixed controls, alongside final/minimum separation and termination reason. One 12,000-step run (200 simulated seconds) reduced separation RMSE from 1,533.7 ft to 881.1 ft, a 42.6% reduction. The controller ended at 2,368.0 ft (minimum 2,351.8 ft), so it improved tracking over fixed controls but did not reach the 1,500 ft target. This is a deterministic proportional-control baseline result, not an RL result or a statistical confidence bound. Change the target gap, initial gap, duration or public JSBSim script through `TwoAircraftConfig` or the CLI.

Run the tests with:

```bash
python -m pytest -q
ruff check .
```

## Observation

The public baseline exposes a small vector derived from JSBSim properties:

- altitude
- calibrated airspeed
- roll
- pitch
- true heading
- body roll/pitch/yaw rates

Values are scaled to roughly comparable numeric ranges before being returned to an RL policy.

## Action

A continuous four-dimensional vector in `[-1, 1]` controls:

```text
[elevator, aileron, rudder, throttle]
```

Throttle is mapped from `[-1, 1]` into `[0, 1]` before being written to the JSBSim property tree.

## Reward

The included reward is intentionally generic and non-tactical: remain near a target altitude while avoiding extreme roll/pitch angles. It exists only to make the environment executable as an RL sandbox.

## Next steps

- Gymnasium adapter
- PPO/SAC baselines
- multi-seed evaluation
- public flight-envelope tasks
- trim-state initialization
- observation/action wrappers
- richer telemetry plots

## License / attribution

JSBSim is a separate open-source project maintained by the JSBSim team. This repository depends on the public Python package but does not vendor or claim ownership of JSBSim aircraft models or source code.
