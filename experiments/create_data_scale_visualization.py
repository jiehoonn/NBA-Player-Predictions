#!/usr/bin/env python3
"""
Generate data scale comparison visualization.

Compares:
- 3 seasons (2022-25), 120 players → 23,325 games
- 5 seasons (2020-25), 200 players → 56,812 games
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from results/data_scale_comparison.md
data = {
    'PTS': {
        '3_seasons': 5.774,
        '5_seasons': 5.454,
        'improvement': -0.320,
        'goal': 3.6
    },
    'REB': {
        '3_seasons': 2.185,
        '5_seasons': 2.134,
        'improvement': -0.051,
        'goal': 2.2
    },
    'AST': {
        '3_seasons': 1.762,
        '5_seasons': 1.647,
        'improvement': -0.115,
        'goal': 2.0
    }
}

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: MAE Comparison
targets = ['PTS', 'REB', 'AST']
x = np.arange(len(targets))
width = 0.35

three_seasons = [data[t]['3_seasons'] for t in targets]
five_seasons = [data[t]['5_seasons'] for t in targets]
goals = [data[t]['goal'] for t in targets]

bars1 = ax1.bar(x - width/2, three_seasons, width, label='3 Seasons (120 players)',
                color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x + width/2, five_seasons, width, label='5 Seasons (200 players)',
                color='#4ECDC4', alpha=0.8)

# Add goal lines
for i, (target, goal) in enumerate(zip(targets, goals)):
    ax1.plot([i-0.5, i+0.5], [goal, goal], 'k--', linewidth=2, alpha=0.5)
    ax1.text(i, goal, f' Goal: {goal}', fontsize=9, va='bottom')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Target Statistic', fontsize=12, fontweight='bold')
ax1.set_title('Performance Comparison: 3 vs 5 Seasons', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(targets)
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: Improvement Breakdown
improvements = [data[t]['improvement'] for t in targets]
colors = ['#51CF66' if imp < 0 else '#FF6B6B' for imp in improvements]

bars = ax2.bar(targets, improvements, color=colors, alpha=0.8)

# Add value labels
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{imp:.3f}\n({imp/data[targets[i]]["3_seasons"]*100:.1f}%)',
            ha='center', va='bottom' if height < 0 else 'top', fontsize=10, fontweight='bold')

ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_ylabel('MAE Improvement (Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Target Statistic', fontsize=12, fontweight='bold')
ax2.set_title('Improvement from Data Scaling', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add dataset info
fig.text(0.5, 0.02,
         'Configuration A: 3 seasons (2022-25), 120 players, 23,325 games | '
         'Configuration B: 5 seasons (2020-25), 200 players, 56,812 games (+144% more data)',
         ha='center', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('results/experiments/data_scale_effect.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved to: results/experiments/data_scale_effect.png")

# Print summary
print("\n" + "="*60)
print("DATA SCALE COMPARISON SUMMARY")
print("="*60)
print("\nDataset Configurations:")
print("  3 Seasons: 2022-25, 120 players, 23,325 games")
print("  5 Seasons: 2020-25, 200 players, 56,812 games (+144% data)")
print("\nPerformance Results:")
for target in targets:
    d = data[target]
    print(f"\n{target}:")
    print(f"  3 Seasons: {d['3_seasons']:.3f} MAE")
    print(f"  5 Seasons: {d['5_seasons']:.3f} MAE")
    print(f"  Improvement: {d['improvement']:.3f} ({d['improvement']/d['3_seasons']*100:.1f}%)")
    print(f"  Goal: {d['goal']:.1f} MAE")
    status = "✅ ACHIEVED" if d['5_seasons'] < d['goal'] else "❌ Not achieved"
    print(f"  Status: {status}")
