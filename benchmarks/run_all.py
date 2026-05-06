"""
Benchmark Runner for Hybrid IK Ensemble

This module runs comprehensive benchmarks across multiple scenarios
to evaluate IK solver performance.

Author: Research Portfolio
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(f"Project root: {project_root}")

import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict
from tabulate import tabulate

from src.robot.arm_model import RobotArm, IKSolver


@dataclass
class BenchmarkResult:
    """Result of a single benchmark"""
    scenario: str
    target: List[float]
    success: bool
    error: float
    iterations: int
    time_ms: float
    in_workspace: bool


class BenchmarkRunner:
    """Run comprehensive IK benchmarks"""
    
    def __init__(self):
        self.robot = RobotArm()
        self.solver = IKSolver(self.robot, method='damped')
        self.results: List[BenchmarkResult] = []
        
    def load_scenarios(self) -> Dict[str, List[np.ndarray]]:
        """Define benchmark scenarios"""
        return {
            'S1_workspace_center': self._workspace_center(),
            'S2_workspace_edge': self._workspace_edge(),
            'S3_singularity_near': self._near_singularities(),
            'S4_joint_limits_near': self._near_joint_limits(),
            'S5_complex_orientations': self._complex_orientations(),
            'S6_random_valid': self._random_valid(30),
            'S7_boundary': self._workspace_boundary(),
        }
    
    def _workspace_center(self) -> List[np.ndarray]:
        """Center of reachable workspace"""
        return [
            np.array([0.4, 0.0, 0.3]),
            np.array([0.5, 0.0, 0.2]),
            np.array([0.45, 0.1, 0.25]),
            np.array([0.35, -0.1, 0.35]),
            np.array([0.4, 0.05, 0.3]),
        ]
    
    def _workspace_edge(self) -> List[np.ndarray]:
        """Near workspace boundaries"""
        return [
            np.array([0.8, 0.0, 0.1]),
            np.array([0.3, 0.0, 0.7]),
            np.array([0.7, 0.2, 0.2]),
            np.array([0.3, -0.3, 0.3]),
            np.array([0.6, 0.0, 0.5]),
        ]
    
    def _near_singularities(self) -> List[np.ndarray]:
        """Near known singular configurations"""
        return [
            np.array([0.5, 0.0, 0.0]),    # Shoulder singularity
            np.array([0.4, 0.3, 0.0]),       # Elbow singularity  
            np.array([0.35, 0.0, 0.35]),   # Wrist singularity
            np.array([0.6, -0.1, 0.1]),   # Near singularity
            np.array([0.45, 0.15, 0.15]),  # Mixed
        ]
    
    def _near_joint_limits(self) -> List[np.ndarray]:
        """Near joint limit boundaries"""
        return [
            np.array([0.75, 0.0, 0.3]),    # Max reach
            np.array([0.3, 0.4, 0.2]),     # Shoulder at limit
            np.array([0.5, 0.0, 0.5]),     # Elbow at limit
            np.array([0.4, -0.3, 0.4]),   # Combined limits
            np.array([0.55, 0.1, 0.35]),   # Near limits
        ]
    
    def _complex_orientations(self) -> List[np.ndarray]:
        """Various workspace positions requiring different configs"""
        return [
            np.array([0.3, 0.3, 0.2]),
            np.array([0.5, -0.2, 0.4]),
            np.array([0.35, 0.15, 0.5]),
            np.array([0.65, 0.05, 0.15]),
            np.array([0.4, -0.15, 0.3]),
        ]
    
    def _workspace_boundary(self) -> List[np.ndarray]:
        """Workspace boundary positions"""
        return [
            np.array([0.82, 0.0, 0.05]),
            np.array([0.27, 0.0, 0.05]),
            np.array([0.55, 0.55, 0.1]),
            np.array([0.55, -0.55, 0.1]),
            np.array([0.4, 0.0, 0.75]),
        ]
    
    def _random_valid(self, n: int) -> List[np.ndarray]:
        """Generate random valid workspace positions"""
        targets = []
        np.random.seed(42)
        
        for _ in range(n):
            # Sample random position in cylinder workspace
            r = np.random.uniform(0.3, 0.75)
            theta = np.random.uniform(-np.pi, np.pi)
            z = np.random.uniform(0.05, 0.6)
            targets.append(np.array([r * np.cos(theta), r * np.sin(theta), z]))
        
        return targets
    
    def run_scenario(self, name: str, targets: List[np.ndarray]) -> List[BenchmarkResult]:
        """Run benchmark for a single scenario"""
        results = []
        
        for target in targets:
            start_time = time.time()
            success, solution, error, iterations = self.solver.solve(target)
            elapsed_ms = (time.time() - start_time) * 1000
            
            in_workspace = self.robot.is_in_workspace(target)
            
            result = BenchmarkResult(
                scenario=name,
                target=target.tolist(),
                success=success,
                error=error,
                iterations=iterations,
                time_ms=elapsed_ms,
                in_workspace=in_workspace
            )
            results.append(result)
        
        return results
    
    def run_all(self) -> Dict:
        """Run all benchmark scenarios"""
        print("Running IK benchmarks...")
        
        scenarios = self.load_scenarios()
        
        for name, targets in scenarios.items():
            print(f"  {name}: {len(targets)} targets...")
            results = self.run_scenario(name, targets)
            self.results.extend(results)
        
        return self.summarize()
    
    def summarize(self) -> Dict:
        """Generate summary statistics"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        
        times = [r.time_ms for r in self.results]
        errors = [r.error for r in self.results]
        
        by_scenario = {}
        for r in self.results:
            if r.scenario not in by_scenario:
                by_scenario[r.scenario] = {'success': 0, 'total': 0, 'times': [], 'errors': []}
            by_scenario[r.scenario]['total'] += 1
            if r.success:
                by_scenario[r.scenario]['success'] += 1
            by_scenario[r.scenario]['times'].append(r.time_ms)
            by_scenario[r.scenario]['errors'].append(r.error)
        
        # Calculate per-scenario stats
        scenario_stats = []
        for name, data in by_scenario.items():
            scenario_stats.append({
                'Scenario': name,
                'Success Rate': f"{data['success']}/{data['total']} ({100*data['success']/data['total']:.0f}%)",
                'Avg Time (ms)': f"{np.mean(data['times']):.1f}",
                'Avg Error': f"{np.mean(data['errors']):.4f}"
            })
        
        return {
            'overall': {
                'total_runs': total,
                'success_rate': f"{successful}/{total} ({100*successful/total:.1f}%)",
                'avg_time_ms': f"{np.mean(times):.1f}",
                'avg_error': f"{np.mean(errors):.4f}"
            },
            'by_scenario': scenario_stats
        }


def main():
    """Run benchmarks and print results"""
    runner = BenchmarkRunner()
    summary = runner.run_all()
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    print("\n### Overall Performance ###")
    for k, v in summary['overall'].items():
        print(f"  {k}: {v}")
    
    print("\n### By Scenario ###")
    print(tabulate(summary['by_scenario'], headers='keys', tablefmt='grid'))
    
    # Save results
    with open('benchmarks/results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nResults saved to benchmarks/results.json")


if __name__ == "__main__":
    main()