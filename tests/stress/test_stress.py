"""
Stress Tests for Hybrid IK Ensemble

This module implements comprehensive stress testing for IK solvers
under various failure modes, edge conditions, and extreme scenarios.

Author: Research Portfolio
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.robot.arm_model import RobotArm, IKSolver


@dataclass
class StressTestResult:
    """Result of a stress test"""
    test_name: str
    passed: bool
    metric: str
    expected: float
    actual: float
    details: str


class StressTestRunner:
    """Comprehensive stress test runner for IK solvers"""
    
    def __init__(self):
        self.robot = RobotArm()
        self.solver = IKSolver(self.robot, method='damped')
        self.results: List[StressTestResult] = []
    
    def run_all_tests(self) -> List[StressTestResult]:
        """Run all stress tests"""
        print("Running IK Stress Tests...")
        print("="*50)
        
        self.test_time_limits()
        self.test_unreachable_targets()
        self.test_singularity_handling()
        self.test_joint_limit_cycling()
        self.test_numerical_stability()
        self.test_convergence_speed()
        self.test_workspace_boundaries()
        
        return self.results
    
    def test_time_limits(self):
        """Test solver behavior under strict time constraints"""
        print("\n[Test 1] Time Limits")
        
        # Very fast target
        fast_target = np.array([0.4, 0.0, 0.3])
        
        # With very few iterations
        start = time.time()
        success, solution, error, iters = self.solver.solve(
            fast_target, 
            max_iterations=5,
            tolerance=1e-2
        )
        elapsed_ms = (time.time() - start) * 1000
        
        passed = elapsed_ms < 50  # Should complete in under 50ms
        self.results.append(StressTestResult(
            test_name="time_limits",
            passed=passed,
            metric="completion_time",
            expected=50,
            actual=elapsed_ms,
            details=f"Fast target completed in {elapsed_ms:.1f}ms"
        ))
        print(f"  {'PASS' if passed else 'FAIL'}: {elapsed_ms:.1f}ms")
    
    def test_unreachable_targets(self):
        """Test solver behavior for unreachable positions"""
        print("\n[Test 2] Unreachable Targets")
        
        unreachable = [
            np.array([10.0, 0.0, 0.0]),      # Way too far
            np.array([0.0, 0.0, 10.0]),      # Way too high
            np.array([100.0, 100.0, 100.0]), # Extreme
            np.array([-5.0, -5.0, -5.0]),   # Behind base
        ]
        
        failures_handled = 0
        for target in unreachable:
            # Should either fail gracefully or handle error
            try:
                success, solution, error, iters = self.solver.solve(
                    target, 
                    max_iterations=10,
                    tolerance=1e-1
                )
                # It's OK if it fails - just needs to not crash
                if not success:
                    failures_handled += 1
            except:
                failures_handled += 1
        
        passed = failures_handled == len(unreachable)
        self.results.append(StressTestResult(
            test_name="unreachable_handling",
            passed=passed,
            metric="graceful_failure",
            expected=len(unreachable),
            actual=failures_handled,
            details=f"Handled {failures_handled}/{len(unreachable)} unreachable"
        ))
        print(f"  {'PASS' if passed else 'FAIL'}: {failures_handled}/{len(unreachable)} handled gracefully")
    
    def test_singularity_handling(self):
        """Test solver at known singular configurations"""
        print("\n[Test 3] Singularity Handling")
        
        # Near known singularities
        singular_positions = [
            np.array([0.5, 0.0, 0.0]),      # Shoulder singularity
            np.array([0.4, 0.3, 0.0]),      # Elbow singularity
            np.array([0.35, 0.0, 0.35]),   # Wrist singularity
        ]
        
        handled = 0
        for target in singular_positions:
            try:
                success, solution, error, iters = self.solver.solve(target)
                if error < 0.1:  # Acceptable error near singularities
                    handled += 1
            except:
                pass
        
        passed = handled >= len(singular_positions) * 0.6  # At least 60%
        self.results.append(StressTestResult(
            test_name="singularity_handling",
            passed=passed,
            metric="stability",
            expected=0.6,
            actual=handled/len(singular_positions),
            details=f"Handled {handled}/{len(singular_positions)}"
        ))
        print(f"  {'PASS' if passed else 'FAIL'}: {handled}/{len(singular_positions)} stable")
    
    def test_joint_limit_cycling(self):
        """Test solver doesn't get stuck cycling at joint limits"""
        print("\n[Test 4] Joint Limit Cycling")
        
        # Position near limits
        limit_targets = [
            np.array([0.8, 0.0, 0.1]),   # Max reach
            np.array([0.3, 0.4, 0.2]),     # Shoulder limit
            np.array([0.5, 0.0, 0.6]),   # Elbow limit
        ]
        
        stable = 0
        for target in limit_targets:
            success1, sol1, _, _ = self.solver.solve(target)
            success2, sol2, _, _ = self.solver.solve(target)  # Run twice
            
            # Should give consistent results
            if success1 and success2:
                if np.allclose(sol1, sol2, atol=0.1):
                    stable += 1
        
        passed = stable >= len(limit_targets) * 0.7
        self.results.append(StressTestResult(
            test_name="joint_limit_cycling",
            passed=passed,
            metric="consistency",
            expected=0.7,
            actual=stable/len(limit_targets),
            details=f"Consistent {stable}/{len(limit_targets)}"
        ))
        print(f"  {'PASS' if passed else 'FAIL'}: {stable}/{len(limit_targets)} consistent")
    
    def test_numerical_stability(self):
        """Test numerical stability with small perturbations"""
        print("\n[Test 5] Numerical Stability")
        
        base = np.array([0.4, 0.1, 0.25])
        perturbations = [
            base + np.array([0.001, 0, 0]),
            base + np.array([0, 0.001, 0]),
            base + np.array([0, 0, 0.001]),
            base + np.array([0.01, 0.01, 0.01]),
        ]
        
        stable = 0
        for target in perturbations:
            try:
                success, solution, error, iters = self.solver.solve(target)
                if success and error < 0.1:
                    stable += 1
            except:
                pass
        
        passed = stable >= len(perturbations) * 0.75
        self.results.append(StressTestResult(
            test_name="numerical_stability",
            passed=passed,
            metric="reliability",
            expected=0.75,
            actual=stable/len(perturbations),
            details=f"Stable {stable}/{len(perturbations)}"
        ))
        print(f"  {'PASS' if passed else 'FAIL'}: {stable}/{len(perturbations)} stable")
    
    def test_convergence_speed(self):
        """Test convergence speed across workspace"""
        print("\n[Test 6] Convergence Speed")
        
        # Random positions across workspace
        np.random.seed(42)
        targets = []
        for _ in range(20):
            r = np.random.uniform(0.35, 0.7)
            theta = np.random.uniform(-np.pi, np.pi)
            z = np.random.uniform(0.1, 0.5)
            targets.append(np.array([r*np.cos(theta), r*np.sin(theta), z]))
        
        times = []
        for target in targets:
            start = time.time()
            _, _, _, iters = self.solver.solve(target)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        passed = avg_time < 50 and max_time < 200
        self.results.append(StressTestResult(
            test_name="convergence_speed",
            passed=passed,
            metric="speed",
            expected=50,
            actual=avg_time,
            details=f"Avg: {avg_time:.1f}ms, Max: {max_time:.1f}ms"
        ))
        print(f"  {'PASS' if passed else 'FAIL'}: Avg {avg_time:.1f}ms, Max {max_time:.1f}ms")
    
    def test_workspace_boundaries(self):
        """Test at exact workspace boundaries"""
        print("\n[Test 7] Workspace Boundaries")
        
        # Near min and max reach
        boundary_targets = [
            np.array([0.26, 0.0, 0.1]),   # Min reach
            np.array([0.84, 0.0, 0.05]),  # Max reach
            np.array([0.4, 0.4, 0.1]),    # Diagonal
            np.array([0.4, -0.4, 0.1]),   # Other diagonal
        ]
        
        handled = 0
        for target in boundary_targets:
            try:
                success, solution, error, iters = self.solver.solve(target)
                if success or error < 0.2:
                    handled += 1
            except:
                pass
        
        passed = handled >= len(boundary_targets) * 0.5
        self.results.append(StressTestResult(
            test_name="workspace_boundaries",
            passed=passed,
            metric="boundary_handling",
            expected=0.5,
            actual=handled/len(boundary_targets),
            details=f"Handled {handled}/{len(boundary_targets)}"
        ))
        print(f"  {'PASS' if passed else 'FAIL'}: {handled}/{len(boundary_targets)}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*50)
        print("STRESS TEST SUMMARY")
        print("="*50)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"\nPassed: {passed}/{total}")
        print(f"Success Rate: {100*passed/total:.0f}%\n")
        
        for r in self.results:
            status = "✓" if r.passed else "✗"
            print(f"{status} {r.test_name}: {r.details}")
        
        return passed == total


def main():
    """Run all stress tests"""
    runner = StressTestRunner()
    results = runner.run_all_tests()
    all_passed = runner.print_summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())