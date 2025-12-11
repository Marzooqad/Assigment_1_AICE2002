
# Create visualizations

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Baseline vs Tuned F1-macro comparison
ax1 = axes[0]
x_pos = np.arange(len(improvements))
width = 0.35

baseline_scores = [imp['Baseline_F1'] for imp in improvements]
tuned_scores = [imp['Tuned_F1'] for imp in improvements]
labels = [f"{imp['Model'][:3]}\n{imp['Target'].split('_')[1]}" for imp in improvements]

ax1.bar(x_pos - width/2, baseline_scores, width, label='Baseline', 
        color='lightcoral', edgecolor='black')
ax1.bar(x_pos + width/2, tuned_scores, width, label='Tuned', 
        color='lightgreen', edgecolor='black')

ax1.set_xlabel('Model and Target', fontweight='bold')
ax1.set_ylabel('F1-Macro Score', fontweight='bold')
ax1.set_title('Baseline vs Tuned Performance', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, max(max(baseline_scores), max(tuned_scores)) * 1.2])

# Plot 2: Percentage improvement
ax2 = axes[1]
pct_improvements = [imp['Pct_Improvement'] for imp in improvements]
colors = ['green' if x > 0 else 'red' for x in pct_improvements]

bars = ax2.bar(x_pos, pct_improvements, color=colors, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_xlabel('Model and Target', fontweight='bold')
ax2.set_ylabel('Improvement (%)', fontweight='bold')
ax2.set_title('Percentage Improvement from Tuning', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels)
ax2.grid(axis='y', alpha=0.3)

for i, (bar, pct) in enumerate(zip(bars, pct_improvements)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{pct:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')

plt.tight_layout()
plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
