
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Data Preprocessing Analysis - Task 2', fontsize=16, fontweight='bold')

# Plot 1: Feature distribution before scaling (sample of 5 features)
ax1 = axes[0, 0]
sample_features = np.random.choice(X_train.shape[1], 5, replace=False)
for i in sample_features:
    ax1.hist(X_train[:, i], bins=30, alpha=0.5, label=f'Feature {i}')
ax1.set_xlabel('Feature Value')
ax1.set_ylabel('Frequency')
ax1.set_title('Feature Distributions Before Scaling', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Feature distribution after scaling (same features)
ax2 = axes[0, 1]
for i in sample_features:
    ax2.hist(X_train_scaled[:, i], bins=30, alpha=0.5, label=f'Feature {i}')
ax2.set_xlabel('Scaled Feature Value')
ax2.set_ylabel('Frequency')
ax2.set_title('Feature Distributions After Scaling', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Mean and std of all features before/after scaling
ax3 = axes[0, 2]
feature_means_before = X_train.mean(axis=0)
feature_stds_before = X_train.std(axis=0)
feature_means_after = X_train_scaled.mean(axis=0)
feature_stds_after = X_train_scaled.std(axis=0)
ax3.scatter(range(len(feature_means_before)), feature_means_before, 
            alpha=0.3, label='Mean (before)', s=10)
ax3.scatter(range(len(feature_means_after)), feature_means_after, 
            alpha=0.3, label='Mean (after)', s=10)
ax3.set_xlabel('Feature Index')
ax3.set_ylabel('Mean Value')
ax3.set_title('Feature Means Before/After Scaling', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Class distribution for target_human
ax4 = axes[1, 0]
labels_human = list(train_human_dist.keys())
counts_human = list(train_human_dist.values())
bars = ax4.bar(range(len(labels_human)), counts_human, edgecolor='black')
ax4.set_xticks(range(len(labels_human)))
ax4.set_xticklabels(labels_human, rotation=45)
ax4.set_ylabel('Number of Samples')
ax4.set_title('target_human Training Set Distribution', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for i, (label, count) in enumerate(zip(labels_human, counts_human)):
    ax4.text(i, count + 30, str(count), ha='center', va='bottom', fontweight='bold')

# Plot 5: Class distribution for target_asv
ax5 = axes[1, 1]
labels_asv = list(train_asv_dist.keys())
counts_asv = list(train_asv_dist.values())
bars = ax5.bar(range(len(labels_asv)), counts_asv, edgecolor='black', color='coral')
ax5.set_xticks(range(len(labels_asv)))
ax5.set_xticklabels(labels_asv, rotation=45)
ax5.set_ylabel('Number of Samples')
ax5.set_title('target_asv Training Set Distribution', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
for i, (label, count) in enumerate(zip(labels_asv, counts_asv)):
    ax5.text(i, count + 30, str(count), ha='center', va='bottom', fontweight='bold')

# Plot 6: Imbalance ratio comparison
ax6 = axes[1, 2]
ratios = [human_ratio, asv_ratio]
names = ['target_human', 'target_asv']
colors = ['skyblue', 'coral']
bars = ax6.bar(range(2), ratios, color=colors, edgecolor='black')
ax6.set_xticks(range(2))
ax6.set_xticklabels(names)
ax6.set_ylabel('Imbalance Ratio')
ax6.set_title('Class Imbalance Comparison', fontweight='bold')
ax6.axhline(y=2.0, color='orange', linestyle='--', label='Moderate threshold')
ax6.axhline(y=5.0, color='red', linestyle='--', label='Severe threshold')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)
for i, ratio in enumerate(ratios):
    ax6.text(i, ratio + 0.5, f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('step3_preprocessing.png', dpi=300, bbox_inches='tight')
print("Visualization saved: step3_preprocessing.png")

print("Step 3 complete. Data ready for model training.")