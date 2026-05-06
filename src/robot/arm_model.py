"""
Robotic Arm Model with proper Denavit-Hartenberg kinematics

This module implements a complete 6-DOF industrial robot arm model
using standard DH parameters.

Author: Research Portfolio
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import math


@dataclass
class DHParmas:
    """Denavit-Hartenberg parameters"""
    theta: float  # Joint angle
    d: float      # Link offset
    a: float      # Link length
    alpha: float  # Link twist


class RobotArm:
    """6-DOF Industrial Robot Arm (UR5-like)"""
    
    def __init__(self):
        # DH parameters for UR5-like 6-DOF arm
        # Order: base, shoulder, elbow, wrist1, wrist2, wrist3
        self.dh_params = [
            DHParmas(theta=0,    d=0.0892,  a=0,       alpha=math.pi/2),  # Base
            DHParmas(theta=0,    d=0,       a=-0.425,  alpha=0),          # Shoulder
            DHParmas(theta=0,    d=0,       a=-0.392,  alpha=0),          # Elbow
            DHParmas(theta=0,    d=0.10915, a=0,       alpha=math.pi/2),  # Wrist1
            DHParmas(theta=0,    d=0.09465, a=0,       alpha=-math.pi/2), # Wrist2
            DHParmas(theta=0,    d=0.0823,  a=0,       alpha=0)           # Wrist3
        ]
        
        # Joint limits [rad]
        self.joint_limits = np.array([
            [-math.pi, math.pi],       # Joint 1: -180 to 180
            [-math.pi, math.pi],       # Joint 2: -180 to 180
            [-math.pi, math.pi],       # Joint 3: -180 to 180
            [-math.pi*2, math.pi*2],   # Joint 4: -360 to 360
            [-math.pi*2, math.pi*2],   # Joint 5: -360 to 360
            [-math.pi*2, math.pi*2]    # Joint 6: -360 to 360
        ])
        
        # Workspace bounds
        self.workspace_min = 0.26  # Minimum reach
        self.workspace_max = 0.85  # Maximum reach
        
    def dh_transform(self, dh: DHParmas) -> np.ndarray:
        """Compute DH transformation matrix"""
        ct = math.cos(dh.theta)
        st = math.sin(dh.theta)
        cd = math.cos(dh.d)
        sd = math.sin(dh.d)
        ca = math.cos(dh.alpha)
        sa = math.sin(dh.alpha)
        
        # Transformation matrix
        return np.array([
            [ct, -st*ca,  st*sa,  dh.a*ct],
            [st,  ct*ca,  -ct*sa, dh.a*st],
            [0,   sa,     ca,     dh.d],
            [0,   0,      0,      1]
        ])
    
    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics - joint angles to end-effector pose
        
        Args:
            joints: 6 joint angles in radians
            
        Returns:
            [x, y, z] position of end-effector
        """
        # Update DH parameters with joint angles
        T = np.eye(4)
        
        for i, dh in enumerate(self.dh_params):
            # Create copy with current joint angle
            dh_i = DHParmas(
                theta=dh.theta + joints[i],
                d=dh.d,
                a=dh.a,
                alpha=dh.alpha
            )
            T = T @ self.dh_transform(dh_i)
        
        # Extract position
        return T[:3, 3]
    
    def jacobian(self, joints: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Compute geometric Jacobian numerically"""
        pos_current = self.forward_kinematics(joints)
        J = np.zeros((3, 6))
        
        for i in range(6):
            joints_plus = joints.copy()
            joints_plus[i] += epsilon
            pos_plus = self.forward_kinematics(joints_plus)
            J[:, i] = (pos_plus - pos_current) / epsilon
            
        return J
    
    def is_valid_position(self, joints: np.ndarray) -> bool:
        """Check if position is within joint limits"""
        for i, (q, limits) in enumerate(zip(joints, self.joint_limits)):
            if q < limits[0] or q > limits[1]:
                return False
        return True
    
    def is_in_workspace(self, position: np.ndarray) -> bool:
        """Check if position is within reachable workspace"""
        distance = np.linalg.norm(position)
        return self.workspace_min <= distance <= self.workspace_max
    
    def compute_singularity_distance(self, joints: np.ndarray) -> float:
        """Compute minimum singular value of Jacobian (singularity measure)"""
        try:
            J = self.jacobian(joints)
            _, s, _ = np.linalg.svd(J)
            return s[-1] if len(s) > 0 else 0.0
        except:
            return 0.0


class IKSolver:
    """Iterative IK solver using Jacobian methods"""
    
    def __init__(self, robot: RobotArm, method: str = 'damped'):
        self.robot = robot
        self.method = method
        self.damping = 0.1  # Damping factor for DLS
        
    def solve(self, 
              target: np.ndarray,
              initial_guess: Optional[np.ndarray] = None,
              max_iterations: int = 100,
              tolerance: float = 1e-4) -> Tuple[bool, np.ndarray, float, int]:
        """
        Solve inverse kinematics
        
        Returns:
            (success, solution, error, iterations)
        """
        if initial_guess is None:
            initial_guess = np.zeros(6)
            
        q = initial_guess.copy()
        
        for iteration in range(max_iterations):
            # Compute current position
            current_pos = self.robot.forward_kinematics(q)
            
            # Error
            error = target - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < tolerance:
                return True, q, error_norm, iteration + 1
            
            # Compute Jacobian
            J = self.robot.jacobian(q)
            
            # Solve using selected method
            if self.method == 'damped':
                # Damped Least Squares
                JJT = J @ J.T
                JJT_reg = JJT + self.damping**2 * np.eye(3)
                delta_q = J.T @ np.linalg.solve(JJT_reg, error)
            elif self.method == 'transpose':
                # Jacobian transpose
                delta_q = 0.5 * J.T @ error
            elif self.method == 'pseudo':
                # Pseudoinverse
                pinv = np.linalg.pinv(J)
                delta_q = pinv @ error
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Update joints
            q = q + delta_q
            
            # Apply joint limits
            q = np.clip(q, self.robot.joint_limits[:, 0], self.robot.joint_limits[:, 1])
        
        # Final error
        final_pos = self.robot.forward_kinematics(q)
        error = np.linalg.norm(target - final_pos)
        
        return error < 0.01, q, error, max_iterations


def test_robot():
    """Test the robot model"""
    robot = RobotArm()
    
    print("=== Robot Arm Model Test ===")
    
    # Test FK at home position
    home = np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0])
    pos = robot.forward_kinematics(home)
    print(f"Home position: {pos}")
    
    # Test at different configurations
    configs = [
        (0, 0, 0, 0, 0, 0),           # All zero
        (np.pi/4, 0, 0, 0, 0, 0),       # Rotate base
        (0, np.pi/4, 0, 0, 0, 0),       # Shoulder
        (np.pi/2, -np.pi/4, np.pi/2, 0, np.pi/4, 0),  # Reach forward
    ]
    
    print("\nTest configurations:")
    for i, config in enumerate(configs):
        joints = np.array(config)
        pos = robot.forward_kinematics(joints)
        valid = robot.is_valid_position(joints)
        in_ws = robot.is_in_workspace(pos)
        print(f"Config {i+1}: pos={pos[:3].round(3)}, valid={valid}, in_workspace={in_ws}")
    
    # Test IK
    print("\n=== IK Solver Test ===")
    solver = IKSolver(robot, method='damped')
    
    test_targets = [
        np.array([0.3, 0.0, 0.5]),
        np.array([0.5, 0.2, 0.3]),
        np.array([0.4, -0.2, 0.4]),
    ]
    
    for target in test_targets:
        success, solution, error, iterations = solver.solve(target)
        print(f"Target: {target}")
        print(f"  Success: {success}, Error: {error:.4f}, Iterations: {iterations}")
        if success:
            fk = robot.forward_kinematics(solution)
            print(f"  Solution FK: {fk}")
        print()


if __name__ == "__main__":
    test_robot()