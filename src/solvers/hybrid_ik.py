"""
Hybrid IK Ensemble with Uncertainty Quantification

This module implements a hybrid inverse kinematics system that combines
multiple IK solvers with learned weighting and uncertainty bounds.

Author: Research Portfolio
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IKSolution:
    """Represents an IK solution from a single solver"""
    joints: np.ndarray
    success: bool
    converged: bool
    iterations: int
    uncertainty: float
    solver_name: str
    computation_time: float
    error: float = 0.0


@dataclass
class HybridIKResult:
    """Combined result from ensemble"""
    joints: np.ndarray
    success: bool
    confidence: float
    solver_contributions: dict
    uncertainty_bounds: Tuple[float, float]


@dataclass
class IKConfig:
    """Configuration for IK solvers"""
    max_iterations: int = 100
    tolerance: float = 1e-6
    step_size: float = 0.01
    learning_rate: float = 0.01
    use_neural: bool = True
    ensemble_enabled: bool = True


class RoboticArmModel:
    """Simple 6-DOF robotic arm model"""
    
    def __init__(self, dh_params: Optional[dict] = None):
        self.dh_params = dh_params or self._default_dh()
        self.n_joints = 6
        self.joint_limits = np.array([
            [-np.pi, np.pi],
            [-np.pi/2, np.pi/2],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi/2, np.pi/2],
            [-np.pi, np.pi]
        ])
    
    def _default_dh(self) -> dict:
        return {
            'd1': 0.2435,
            'a2': -0.3125,
            'a3': 0.0,
            'd4': 0.2125,
            'd5': 0.0,
            'd6': 0.15
        }
    
    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """Compute forward kinematics - returns end-effector position"""
        q = joints
        
        # Simplified forward kinematics for 6-DOF arm
        # In practice, use full DH transformation matrix
        x = (np.cos(q[0]) * (self.dh_params['d4'] + self.dh_params['d6'] + 
              np.sin(q[2] + q[3]) * (self.dh_params['a2'] + np.cos(q[1]) * self.dh_params['d1'])))
        y = np.sin(q[0]) * (self.dh_params['d4'] + self.dh_params['d6'] +
              np.sin(q[2] + q[3]) * (self.dh_params['a2'] + np.cos(q[1]) * self.dh_params['d1']))
        z = self.dh_params['d1'] + self.dh_params['d4'] + self.dh_params['d6'] - \
            np.cos(q[2]) * (self.dh_params['a2'] + np.cos(q[1]) * self.dh_params['d1'])
        
        return np.array([x, y, z])
    
    def jacobian(self, joints: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix"""
        # Numerical Jacobian
        eps = 1e-6
        J = np.zeros((3, self.n_joints))
        f_current = self.forward_kinematics(joints)
        
        for i in range(self.n_joints):
            joints_plus = joints.copy()
            joints_plus[i] += eps
            J[:, i] = (self.forward_kinematics(joints_plus) - f_current) / eps
        
        return J
    
    def is_in_limits(self, joints: np.ndarray) -> bool:
        """Check if joints are within limits"""
        for i, (q, limits) in enumerate(zip(joints, self.joint_limits)):
            if q < limits[0] or q > limits[1]:
                return False
        return True
    
    @property
    def jh_limits(self):
        return self.joint_limits


class AnalyticalSolver:
    """Closed-form analytical IK solver"""
    
    def __init__(self, arm_model: RoboticArmModel):
        self.arm = arm_model
    
    def solve(self, target: np.ndarray, 
              initial_guess: Optional[np.ndarray] = None) -> IKSolution:
        """Solve IK analytically"""
        import time
        start_time = time.time()
        
        if initial_guess is None:
            initial_guess = np.zeros(6)
        
        # Simplified analytical solution for demonstration
        # In practice, implement full geometric solution based on arm geometry
        
        # Basic heuristic for 6-DOF arm
        q = initial_guess.copy()
        
        # Set first joint to point toward target
        if np.linalg.norm(target[:2]) > 1e-6:
            q[0] = np.arctan2(target[1], target[0])
        
        # Distance-based solution for remaining joints
        distance = np.linalg.norm(target)
        
        # Approximate solution
        if distance > 0.1 and distance < 2.0:
            q[1] = np.clip(distance - 0.5, -np.pi/3, np.pi/3)
            q[2] = np.clip(0.5 - distance/4, -np.pi/2, np.pi/2)
            q[3] = np.clip(target[2] / (distance + 0.1), -np.pi/2, np.pi/2)
        
        # Calculate error
        fk_result = self.arm.forward_kinematics(q)
        error = np.linalg.norm(target - fk_result)
        
        success = error < 0.1
        
        computation_time = time.time() - start_time
        
        return IKSolution(
            joints=q,
            success=success,
            converged=success,
            iterations=1,
            uncertainty=0.5 if success else 1.0,
            solver_name="analytical",
            computation_time=computation_time,
            error=error
        )


class JacobianTransposeSolver:
    """Iterative Jacobian transpose IK solver"""
    
    def __init__(self, arm_model: RoboticArmModel, config: IKConfig):
        self.arm = arm_model
        self.config = config
    
    def solve(self, target: np.ndarray,
              initial_guess: Optional[np.ndarray] = None) -> IKSolution:
        """Solve IK using Jacobian transpose method"""
        import time
        start_time = time.time()
        
        if initial_guess is None:
            initial_guess = np.zeros(6)
        
        q = initial_guess.copy()
        
        for iteration in range(self.config.max_iterations):
            current_pos = self.arm.forward_kinematics(q)
            error = target - current_pos
            
            if np.linalg.norm(error) < self.config.tolerance:
                break
            
            J = self.arm.jacobian(q)
            
            try:
                # Pseudoinverse solution
                delta_q = self.config.step_size * J.T @ np.linalg.pinv(J @ J.T) @ error
            except:
                # Fallback to transpose
                delta_q = self.config.step_size * J.T @ error
            
            q = q + delta_q
            
            # Apply joint limits
            q = np.clip(q, self.arm.joint_limits[:, 0], self.arm.joint_limits[:, 1])
        
        final_pos = self.arm.forward_kinematics(q)
        error = np.linalg.norm(target - final_pos)
        
        success = error < 0.05
        computation_time = time.time() - start_time
        
        # Uncertainty based on convergence and iterations
        uncertainty = min(1.0, iteration / self.config.max_iterations) if success else 1.0
        
        return IKSolution(
            joints=q,
            success=success,
            converged=success,
            iterations=iteration + 1,
            uncertainty=uncertainty,
            solver_name="jacobian",
            computation_time=computation_time,
            error=error
        )


class NeuralSolver:
    """Neural network based IK solver"""
    
    def __init__(self, arm_model: RoboticArmModel):
        self.arm = arm_model
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize simple neural network"""
        try:
            import torch
            import torch.nn as nn
            
            self.torch = torch
            
            class SimpleIKNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(3, 64),
                        nn.ReLU(),
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 6)
                    )
                
                def forward(self, x):
                    return torch.tanh(self.net(x)) * np.pi
                    
            self.model = SimpleIKNet()
            self.model.eval()
            
            # Initialize with random weights (in practice, train first)
            logger.info("Neural solver initialized (untrained)")
            
        except ImportError:
            logger.warning("PyTorch not available, neural solver disabled")
            self.model = None
    
    def solve(self, target: np.ndarray,
              initial_guess: Optional[np.ndarray] = None) -> IKSolution:
        """Solve IK using neural network"""
        import time
        start_time = time.time()
        
        if self.model is None:
            return IKSolution(
                joints=np.zeros(6),
                success=False,
                converged=False,
                iterations=0,
                uncertainty=1.0,
                solver_name="neural",
                computation_time=time.time() - start_time,
                error=1.0
            )
        
        with self.torch.no_grad():
            target_tensor = self.torch.tensor(target[:3], dtype=self.torch.float32)
            q_pred = self.model(target_tensor).numpy()
        
        fk_result = self.arm.forward_kinematics(q_pred)
        error = np.linalg.norm(target - fk_result)
        
        # Uncertainty is higher for untrained network
        uncertainty = 0.7
        
        return IKSolution(
            joints=q_pred,
            success=error < 0.2,
            converged=True,
            iterations=1,
            uncertainty=uncertainty,
            solver_name="neural",
            computation_time=time.time() - start_time,
            error=error
        )


class HybridIKEnsemble:
    """Main hybrid IK ensemble solver"""
    
    def __init__(self, config: Optional[IKConfig] = None):
        self.config = config or IKConfig()
        self.arm = RoboticArmModel()
        
        # Initialize solvers
        self.solvers = {
            'analytical': AnalyticalSolver(self.arm),
            'jacobian': JacobianTransposeSolver(self.arm, self.config),
            'neural': NeuralSolver(self.arm)
        }
        
        self.weights = {
            'analytical': 0.3,
            'jacobian': 0.4,
            'neural': 0.3
        }
        
        logger.info("Hybrid IK Ensemble initialized")
    
    def solve(self, target: np.ndarray,
              initial_guess: Optional[np.ndarray] = None) -> HybridIKResult:
        """Solve IK using ensemble of solvers"""
        
        results = {}
        
        for name, solver in self.solvers.items():
            results[name] = solver.solve(target, initial_guess)
        
        # Combine results
        if self.config.ensemble_enabled:
            return self._combine_solutions(results)
        else:
            # Use best individual solver
            best = min(results.values(), key=lambda x: x.error)
            return HybridIKResult(
                joints=best.joints,
                success=best.success,
                confidence=1.0 - best.uncertainty,
                solver_contributions={best.solver_name: 1.0},
                uncertainty_bounds=(best.uncertainty, 1.0)
            )
    
    def _combine_solutions(self, results: dict) -> HybridIKResult:
        """Combine solutions using weighted average"""
        
        # Filter successful solutions
        successful = {k: v for k, v in results.items() if v.success}
        
        if not successful:
            # All failed - return best effort
            best = min(results.values(), key=lambda x: x.error)
            return HybridIKResult(
                joints=best.joints,
                success=False,
                confidence=0.0,
                solver_contributions={'failed': 1.0},
                uncertainty_bounds=(1.0, 1.0)
            )
        
        # Weighted combination
        total_weight = 0
        combined_joints = np.zeros(6)
        contributions = {}
        
        for name, result in successful.items():
            weight = self.weights.get(name, 0.33)
            combined_joints += weight * result.joints
            contributions[name] = weight
            total_weight += weight
        
        if total_weight > 0:
            combined_joints /= total_weight
        
        # Calculate confidence
        avg_uncertainty = np.mean([r.uncertainty for r in successful.values()])
        confidence = 1.0 - avg_uncertainty
        
        # Uncertainty bounds
        uncertainties = [r.uncertainty for r in successful.values()]
        uncertainty_bounds = (min(uncertainties), max(uncertainties))
        
        return HybridIKResult(
            joints=combined_joints,
            success=True,
            confidence=confidence,
            solver_contributions=contributions,
            uncertainty_bounds=uncertainty_bounds
        )


def solve_ik(target: np.ndarray, 
             method: str = "ensemble",
             config: Optional[IKConfig] = None) -> HybridIKResult:
    """
    Main entry point for IK solving.
    
    Args:
        target: Target end-effector position [x, y, z]
        method: "ensemble", "analytical", "jacobian", or "neural"
        config: Optional configuration
    
    Returns:
        HybridIKResult with solution and uncertainty
    """
    if config is None:
        config = IKConfig()
    
    solver = HybridIKEnsemble(config)
    
    if method == "ensemble":
        return solver.solve(target)
    elif method in solver.solvers:
        result = solver.solvers[method].solve(target)
        return HybridIKResult(
            joints=result.joints,
            success=result.success,
            confidence=1.0 - result.uncertainty,
            solver_contributions={method: 1.0},
            uncertainty_bounds=(result.uncertainty, 1.0)
        )
    else:
        raise ValueError(f"Unknown method: {method}")