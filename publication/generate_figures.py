"""
Generate figures for paper: system diagram and benchmark plots
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Figure 1: System Architecture Diagram
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 8)
ax1.axis('off')
ax1.set_title('Hybrid IK Ensemble System Architecture', fontsize=14, fontweight='bold')

# Input
input_box = mpatches.FancyBboxPatch((1, 6), 2.5, 1, boxstyle="round,pad=0.05", 
                                      facecolor='#E3F2FD', edgecolor='black', linewidth=2)
ax1.add_patch(input_box)
ax1.text(2.25, 6.5, 'Target\n(x, y, z)', ha='center', va='center', fontsize=10)

# Arrow to solvers
ax1.annotate('', xy=(4.5, 6.5), xytext=(3.5, 6.5), arrowprops=dict(arrowstyle='->', lw=2))

# Solver boxes
# DLS Solver
dls_box = mpatches.FancyBboxPatch((4.5, 5.2), 2.5, 1.5, boxstyle="round,pad=0.05",
                                   facecolor='#C8E6C9', edgecolor='black', linewidth=2)
ax1.add_patch(dls_box)
ax1.text(5.75, 6.0, 'Damped Least\nSquares (DLS)', ha='center', va='center', fontsize=10)
ax1.text(5.75, 5.5, 'λ=0.01, 100 iters', ha='center', va='center', fontsize=8, color='gray')

# NN Solver
nn_box = mpatches.FancyBboxPatch((4.5, 3.2), 2.5, 1.5, boxstyle="round,pad=0.05",
                                  facecolor='#FFCCBC', edgecolor='black', linewidth=2)
ax1.add_patch(nn_box)
ax1.text(5.75, 4.0, 'Neural Network\nSolver (MLP)', ha='center', va='center', fontsize=10)
ax1.text(5.75, 3.5, '3→128→128→64→6', ha='center', va='center', fontsize=8, color='gray')

# Arrows to ensemble
ax1.annotate('', xy=(7.5, 5.95), xytext=(7, 5.95), arrowprops=dict(arrowstyle='->', lw=2))
ax1.annotate('', xy=(7.5, 3.95), xytext=(7, 3.95), arrowprops=dict(arrowstyle='->', lw=2))

# Ensemble
ens_box = mpatches.FancyBboxPatch((8, 4.5), 2.5, 2, boxstyle="round,pad=0.05",
                                   facecolor='#FFF9C4', edgecolor='black', linewidth=2)
ax1.add_patch(ens_box)
ax1.text(9.25, 5.7, 'Ensemble Weighting', ha='center', va='center', fontsize=11, fontweight='bold')
ax1.text(9.25, 5.2, 'Learned weights', ha='center', va='center', fontsize=9)
ax1.text(9.25, 4.7, 'DLS=0.45, NN=0.55', ha='center', va='center', fontsize=9, color='gray')

# Arrow to output
ax1.annotate('', xy=(11.5, 5.5), xytext=(10.5, 5.5), arrowprops=dict(arrowstyle='->', lw=2))

# Output
out_box = mpatches.FancyBboxPatch((11.5, 4.8), 1.2, 1.4, boxstyle="round,pad=0.05",
                                   facecolor='#E1BEE7', edgecolor='black', linewidth=2)
ax1.add_patch(out_box)
ax1.text(12.1, 5.7, 'Joint\nAngles', ha='center', va='center', fontsize=10)

# Legend
ax1.text(1, 1.5, 'Components:', fontsize=10, fontweight='bold')
ax1.add_patch(mpatches.FancyBboxPatch((1, 0.8), 0.4, 0.4, boxstyle="round", 
                                       facecolor='#C8E6C9', edgecolor='black'))
ax1.text(1.5, 1.0, 'Analytical (DLS)', fontsize=9)
ax1.add_patch(mpatches.FancyBboxPatch((4.5, 0.8), 0.4, 0.4, boxstyle="round",
                                       facecolor='#FFCCBC', edgecolor='black'))
ax1.text(5, 1.0, 'Learned (NN)', fontsize=9)
ax1.add_patch(mpatches.FancyBboxPatch((8, 0.8), 0.4, 0.4, boxstyle="round",
                                       facecolor='#FFF9C4', edgecolor='black'))
ax1.text(8.5, 1.0, 'Ensemble', fontsize=9)

plt.tight_layout()
plt.savefig('system_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("System diagram saved to publication/system_diagram.png")

# Figure 2: Benchmark Results Bar Chart
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

scenarios = ['Workspace\nCenter', 'Workspace\nEdge', 'Singularity\nNear', 'Joint\nLimits', 
             'Complex\nOrient.', 'Random\nValid', 'Boundary']
success_rates = [20, 80, 80, 100, 80, 100, 80]
colors = ['#FF6B6B', '#4ECDC4', '#4ECDC4', '#45B7D1', '#4ECDC4', '#45B7D1', '#4ECDC4']

bars = ax2.bar(scenarios, success_rates, color=colors, edgecolor='black', linewidth=1.5)

ax2.set_ylabel('Success Rate (%)', fontsize=12)
ax2.set_xlabel('Benchmark Scenario', fontsize=12)
ax2.set_title('IK Solver Performance by Scenario (60 Test Targets)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.axhline(y=86.7, color='red', linestyle='--', linewidth=2, label='Overall: 86.7%')
ax2.legend(loc='upper right', fontsize=10)

# Add value labels on bars
for bar, rate in zip(bars, success_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{rate}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_plot.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Benchmark plot saved to publication/benchmark_plot.png")

# Figure 3: Comparison plot
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Success rate comparison
methods = ['DLS Only', 'NN Only', 'Hybrid Ensemble']
success = [95, 100, 84]
times = [8.0, 2.4, 9.8]

colors = ['#2196F3', '#4CAF50', '#FF9800']
bars = ax3a.bar(methods, success, color=colors, edgecolor='black', linewidth=1.5)
ax3a.set_ylabel('Success Rate (%)', fontsize=12)
ax3a.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
ax3a.set_ylim(0, 110)
for bar, s in zip(bars, success):
    ax3a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
              f'{s}%', ha='center', fontsize=11, fontweight='bold')

# Right: Solve time comparison
bars = ax3b.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5)
ax3b.set_ylabel('Avg Time (ms)', fontsize=12)
ax3b.set_title('Solve Time Comparison', fontsize=12, fontweight='bold')
ax3b.set_ylim(0, 12)
for bar, t in zip(bars, times):
    ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
              f'{t}ms', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Comparison plot saved to publication/comparison_plot.png")

print("\nAll figures generated successfully!")