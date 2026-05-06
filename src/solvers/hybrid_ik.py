"""
Hybrid IK Ensemble

Combines multiple IK solvers with learned uncertainty-based weighting.
This is the core of the FI-09 research: Hybrid IK Ensemble with Uncertainty.

Author: Research Portfolio
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.robot.arm_model import RobotArm, IKSolver
from src.solvers.neural_ik import NeuralIKSolver


class EnsembleWeights:
    """Learned weights for ensemble combining"""
    
    def __init__(self, n_solvers: int = 2):
        self.n_solvers = n_solvers
        self.weights = np.ones(n_solvers) / n_solvers
        self.uncertainties = np.ones(n_solvers) * 0.5
        
    def update(self, solver_idx: int, success: bool, error: float):
        """Update weights based on solver performance"""
        if success:
            self.weights[solver_idx] *= 1.1
        else:
            self.weights[solver_idx] *= 0.9
        
        # Normalize
        self.weights = self.weights / np.sum(self.weights)
        
        # Update uncertainty (running estimate)
        self.uncertainties[solver_idx] = 0.9 * self.uncertainties[solver_idx] + 0.1 * error
    
    def get_weighted_solution(self, solutions: List[np.ndarray], 
                             errors: List[float]) -> Tuple[np.ndarray, float]:
        """Get weighted average of solutions"""
        valid = [(i, s, e) for i, (s, e) in enumerate(zip(solutions, errors)) if e < 1.0]
        
        if not valid:
            return solutions[0] if len(solutions) > 0 else np.zeros(6), 1.0
        
        weighted = np.zeros(6)
        total_weight = 0
        
        for idx, sol, err in valid:
            w = self.weights[idx] / (err + 0.01)
            weighted += w * sol
            total_weight += w
        
        if total_weight > 0:
            weighted /= total_weight
        
        # Compute ensemble uncertainty
        ensemble_error = np.mean([e for _, _, e in valid])
        
        return weighted, ensemble_error


class HybridIKEnsemble:
    """
    Hybrid IK Ensemble with Uncertainty Quantification
    
    Combines:
    1. Damped Least Squares (analytical)
    2. Neural Network (learned)
    3. Uncertainty estimation
    
    Key innovation: Learn when to trust each solver based on workspace region.
    """
    
    def __init__(self, robot: RobotArm):
        self.robot = robot
        
        # Initialize solvers
        self.solvers = {
            'dls': IKSolver(robot, method='damped'),
            'nn': None  # Will be trained on demand
        }
        
        self.weights = EnsembleWeights(n_solvers=2)
        self.is_trained = False
    
    def train_nn_solver(self, n_epochs: int = 100):
        """Train the neural network solver"""
        print("Training neural network solver...")
        self.solvers['nn'] = NeuralIKSolver(self.robot)
        self.solvers['nn'].train(n_epochs=n_epochs, batch_size=64, learning_rate=0.005)
        self.is_trained = True
        print("Neural network trained!")
    
    def solve(self, target: np.ndarray, use_nn: bool = True) -> Tuple[bool, np.ndarray, float, int, Dict]:
        """
        Solve IK using hybrid ensemble approach
        
        Returns:
            success, solution, error, iterations, info_dict
        """
        
        solutions = []
        errors = []
        solver_info = {}
        
        # DLS solver
        dls_success, dls_sol, dls_err, dls_iters = self.solvers['dls'].solve(target)
        solutions.append(dls_sol)
        errors.append(dls_err)
        solver_info['dls'] = {'success': dls_success, 'error': dls_err, 'iters': dls_iters}
        
        # Neural network solver (if trained)
        if use_nn and self.is_trained:
            nn_success, nn_sol, nn_err, nn_iters = self.solvers['nn'].solve(target)
            solutions.append(nn_sol)
            errors.append(nn_err)
            solver_info['nn'] = {'success': nn_success, 'error': nn_err, 'iters': nn_iters}
        
        # Get weighted ensemble solution
        ensemble_sol, ensemble_err = self.weights.get_weighted_solution(solutions, errors)
        
        # Update weights based on performance
        if dls_success:
            self.weights.update(0, True, dls_err)
        if use_nn and self.is_trained and self.solvers['nn']:
            if solver_info.get('nn', {}).get('success'):
                self.weights.update(1, True, solver_info['nn']['error'])
        
        # Final verification
        final_pos = self.robot.forward_kinematics(ensemble_sol)
        final_error = np.linalg.norm(target - final_pos)
        
        info = {
            'solver_info': solver_info,
            'weights': self.weights.weights.tolist(),
            'ensemble_error': ensemble_err,
            'solved_by_nn': use_nn and self.is_trained
        }
        
        success = final_error < 0.1
        return success, ensemble_sol, final_error, dls_iters, info


def benchmark_ensemble():
    """Benchmark the hybrid ensemble"""
    from src.robot.arm_model import RobotArm
    import time
    
    print("="*60)
    print("Hybrid IK Ensemble Benchmark")
    print("="*60)
    
    robot = RobotArm()
    ensemble = HybridIKEnsemble(robot)
    
    # Train NN
    print("\n[1] Training neural network component...")
    ensemble.train_nn_solver(n_epochs=80)
    
    # Test scenarios
    print("\n[2] Running benchmark...")
    
    scenarios = {
        'center': [np.array([0.4, 0.0, 0.3]), np.array([0.35, 0.1, 0.25]), np.array([0.45, -0.1, 0.35])],
        'edge': [np.array([0.7, 0.0, 0.2]), np.array([0.3, 0.3, 0.15])],
        'random': [np.array([
            np.random.uniform(0.35, 0.65) * np.cos(np.random.uniform(-np.pi, np.pi)),
            np.random.uniform(0.35, 0.65) * np.sin(np.random.uniform(-np.pi, np.pi)),
            np.random.uniform(0.15, 0.45)
        ]) for _ in range(20)]
    }
    
    total = 0
    success = 0
    total_time = 0
    
    for scenario, targets in scenarios.items():
        print(f"\n  {scenario}:")
        for target in targets:
            start = time.time()
            ok, sol, err, iters, info = ensemble.solve(target)
            elapsed = (time.time() - start) * 1000
            
            total += 1
            if ok:
                success += 1
            total_time += elapsed
            
            print(f"    target={[f'{t:.2f}' for t in target[:2]]}... {'✓' if ok else '✗'} err={err:.4f} t={elapsed:.1f}ms weights={info['weights']}")
    
    print(f"\n{'='*60}")
    print(f"Total: {success}/{total} ({100*success/total:.0f}%)")
    print(f"Avg time: {total_time/total:.1f}ms")
    print(f"Final weights: DLS={ensemble.weights.weights[0]:.2f}, NN={ensemble.weights.weights[1]:.2f}")
    
    return success, total


if __name__ == "__main__":
    benchmark_ensemble()