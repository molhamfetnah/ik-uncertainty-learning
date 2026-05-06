"""
Neural Network IK Solver

A simple MLP-based inverse kinematics solver using NumPy.
This provides fast approximate solutions that can be refined.

Author: Research Portfolio
"""

import numpy as np
from typing import Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.robot.arm_model import RobotArm


class MLP:
    """Simple Multi-Layer Perceptron implemented in NumPy"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [128, 128, 64]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        self.weights = []
        self.biases = []
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            self.weights.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.biases.append(np.zeros((1, dims[i+1])))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def relu_deriv(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activations = [x]
        self.z_values = []
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ w + b
            self.z_values.append(z)
            
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.tanh(z)  # Output bounded to [-1, 1]
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, y_true: np.ndarray, learning_rate: float):
        m = y_true.shape[0]
        
        # Output layer delta
        delta = (self.activations[-1] - y_true) * (1 - self.activations[-1]**2)
        
        for l in range(len(self.weights) - 1, -1, -1):
            dW = self.activations[l].T @ delta / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if l > 0:
                delta = (delta @ self.weights[l].T) * self.relu_deriv(self.z_values[l-1])
            
            self.weights[l] -= learning_rate * dW
            self.biases[l] -= learning_rate * db
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class NeuralIKSolver:
    """MLP-based IK solver with training capability"""
    
    def __init__(self, robot: RobotArm, hidden_dims: list = [128, 128, 64]):
        self.robot = robot
        self.mlp = MLP(3, 6, hidden_dims)  # 3D target -> 6 joints
        self.is_trained = False
        
        self.joint_limits = np.array([
            [-np.pi, np.pi],
            [-np.pi/2, np.pi/2],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi/2, np.pi/2],
            [-np.pi, np.pi]
        ])
    
    def normalize_target(self, target: np.ndarray) -> np.ndarray:
        """Normalize target to [-1, 1] range based on workspace"""
        norm = np.array([
            0.5,  # x range
            0.5,  # y range  
            0.4   # z range
        ])
        return target / norm
    
    def denormalize_joints(self, joints: np.ndarray) -> np.ndarray:
        """Denormalize from [-1,1] to actual joint limits"""
        joints = np.clip(joints, -1, 1)
        result = np.zeros_like(joints)
        for i in range(6):
            lo, hi = self.joint_limits[i]
            result[i] = lo + (joints[i] + 1) * (hi - lo) / 2
        return result
    
    def generate_training_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data using analytical solver"""
        print(f"Generating {n_samples} training samples...")
        
        targets = []
        solutions = []
        
        # Random valid targets
        np.random.seed(42)
        for _ in range(n_samples):
            r = np.random.uniform(0.3, 0.7)
            theta = np.random.uniform(-np.pi, np.pi)
            z = np.random.uniform(0.1, 0.5)
            target = np.array([r * np.cos(theta), r * np.sin(theta), z])
            
            # Try to solve
            try:
                from src.robot.arm_model import IKSolver
                solver = IKSolver(self.robot, method='damped')
                success, solution, error, _ = solver.solve(target, max_iterations=50)
                
                if success and error < 0.05:
                    targets.append(self.normalize_target(target))
                    
                    # Normalize joints to [-1, 1]
                    normalized = np.zeros(6)
                    for i in range(6):
                        lo, hi = self.joint_limits[i]
                        normalized[i] = 2 * (solution[i] - lo) / (hi - lo) - 1
                    solutions.append(normalized)
            except:
                pass
        
        targets = np.array(targets)
        solutions = np.array(solutions)
        
        print(f"  Generated {len(targets)} valid samples")
        return targets, solutions
    
    def train(self, n_epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the MLP"""
        X, y = self.generate_training_data()
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        print(f"Training MLP for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            
            total_loss = 0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward pass
                predictions = self.mlp.forward(X_batch)
                
                # Compute loss (MSE)
                loss = np.mean((predictions - y_batch) ** 2)
                total_loss += loss
                n_batches += 1
                
                # Backward pass
                self.mlp.backward(y_batch, learning_rate)
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / n_batches
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        print("Training complete!")
    
    def solve(self, target: np.ndarray, max_iterations: int = 10) -> Tuple[bool, np.ndarray, float, int]:
        """Solve IK using neural network prediction + refinement"""
        
        if not self.is_trained:
            return False, np.zeros(6), 1.0, 0
        
        # Normalize input
        target_norm = self.normalize_target(target)
        
        # Forward pass
        joints_norm = self.mlp.predict(target_norm.reshape(1, 3))[0]
        
        # Denormalize
        joints = self.denormalize_joints(joints_norm)
        
        # Refine using one step of gradient descent on error
        for _ in range(max_iterations):
            pos = self.robot.forward_kinematics(joints)
            error = target - pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < 0.01:
                break
            
            # Simple Jacobian-based refinement
            J = self.robot.jacobian(joints)
            try:
                delta_joints = J[:3].T @ np.linalg.inv(J[:3] @ J[:3].T + 0.01 * np.eye(3)) @ error
                joints += delta_joints.flatten() * 0.5
                joints = np.clip(joints, self.joint_limits[:, 0], self.joint_limits[:, 1])
            except:
                pass
        
        # Final error check
        pos = self.robot.forward_kinematics(joints)
        final_error = np.linalg.norm(target - pos)
        
        success = final_error < 0.1
        return success, joints, final_error, max_iterations


def train_and_test():
    """Train and test the neural IK solver"""
    from src.robot.arm_model import IKSolver as AnalyticalSolver
    
    print("="*60)
    print("Neural IK Solver Training & Testing")
    print("="*60)
    
    robot = RobotArm()
    
    # Train neural solver
    nn_solver = NeuralIKSolver(robot)
    nn_solver.train(n_epochs=100, batch_size=64, learning_rate=0.005)
    
    # Test on random targets
    print("\nTesting on random targets...")
    
    test_targets = []
    np.random.seed(123)
    for _ in range(20):
        r = np.random.uniform(0.35, 0.65)
        theta = np.random.uniform(-np.pi, np.pi)
        z = np.random.uniform(0.15, 0.45)
        test_targets.append(np.array([r*np.cos(theta), r*np.sin(theta), z]))
    
    nn_success = 0
    analytic_success = 0
    
    for target in test_targets:
        # Neural solver
        nn_ok, nn_sol, nn_err, _ = nn_solver.solve(target)
        if nn_ok:
            nn_success += 1
        
        # Analytical solver for comparison
        analytic_solver = AnalyticalSolver(robot, method='damped')
        an_ok, an_sol, an_err, _ = analytic_solver.solve(target)
        if an_ok:
            analytic_success += 1
    
    print(f"\nResults:")
    print(f"  Neural Network: {nn_success}/20 ({100*nn_success/20:.0f}%)")
    print(f"  Analytical DLS: {analytic_success}/20 ({100*analytic_success/20:.0f}%)")
    
    return nn_success, analytic_success


if __name__ == "__main__":
    train_and_test()