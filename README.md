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
- Gymnasium adapter with explicit terminated/truncated semantics
- fail-closed action, observation and episode-lifecycle validation
- deterministic random-policy smoke demo
- CSV trajectory export

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python demo.py --steps 800 --seed 42
```

The demo executes a bounded random policy, prints a compact summary and writes a trajectory CSV under `artifacts/`.

## Gymnasium integration

`JSBSimGymEnv` exposes standard `Box` action/observation spaces and the five-value Gymnasium step
contract, so it can be consumed by Gymnasium-compatible PPO/SAC implementations:

```python
import numpy as np

from gym_env import JSBSimGymEnv

env = JSBSimGymEnv()
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(
    np.zeros(4, dtype=np.float32)
)
```

Simulator failure is reported as `terminated`; reaching the configured step budget is reported as
`truncated`. If both happen on the same step, simulator termination takes precedence. The adapter
rejects malformed, non-finite and out-of-range actions, verifies observation/reward contracts, and
requires a new reset after either terminal outcome. A fake core in unit tests exercises these
semantics without pretending to validate aircraft physics.

The reset seed controls Gymnasium and action-space randomness. The bundled JSBSim initial-condition
script itself is deterministic and does not expose a separate stochastic seed through this adapter.

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

- PPO/SAC baselines
- multi-seed evaluation
- public flight-envelope tasks
- trim-state initialization
- observation/action wrappers
- richer telemetry plots

## Validation

GitHub Actions runs Ruff, formatting, tests, compilation and a bounded public JSBSim trajectory on
Python 3.11, 3.12 and 3.13. The trajectory artifact demonstrates executable integration; it is not
a stability, safety or controller-performance claim.

## License / attribution

JSBSim is a separate open-source project maintained by the JSBSim team. This repository depends on the public Python package but does not vendor or claim ownership of JSBSim aircraft models or source code.
