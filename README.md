# Hybrid IK Ensemble with Uncertainty Quantification

## Project Overview

This project implements a hybrid inverse kinematics (IK) system that combines multiple IK solvers with learned weighting and uncertainty quantification for robotic arm control.

## Key Features

- **Hybrid Ensemble**: Combines analytical, Jacobian, and neural network solvers
- **Uncertainty Quantification**: Provides confidence bounds for each solution
- **Adaptive Weighting**: Learns to select best solver based on problem characteristics
- **Comprehensive Testing**: Unit tests, stress tests, and benchmark scenarios

## Architecture

```
Hybrid IK Ensemble
├── Analytical Solver     - Closed-form geometric solution
├── Jacobian Solver      - Iterative gradient descent
├── Neural Solver        - ML-based prediction
└── Ensemble Manager      - Weighted fusion + uncertainty
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from src.solvers.hybrid_ik import solve_ik

# Target position
target = np.array([0.5, 0.3, 0.4])

# Solve using ensemble
result = solve_ik(target, method='ensemble')

print(f"Solution: {result.joints}")
print(f"Confidence: {result.confidence}")
print(f"Uncertainty: {result.uncertainty_bounds}")
```

## Project Structure

```
ik-uncertainty-learning/
├── src/
│   ├── solvers/         # IK solver implementations
│   ├── ensemble/       # Ensemble weighting
│   ├── uncertainty/   # Uncertainty quantification
│   └── robot/         # Robot models
├── tests/             # Test suite
├── benchmarks/       # Evaluation scenarios
└── paper/            # Paper drafts
```

## Status: In Development

This project is currently being implemented. See SPEC.md for detailed specification.

## License

CC BY-NC-ND 4.0 - See LICENSE file