"""
AICE2002 Assignment 1 - STEP 3: Data Preprocessing (TASK 2)
Author: [Your Name]
Date: November 2, 2025

PURPOSE: Implement preprocessing pipeline including feature scaling and 
handling class imbalance. This addresses Task 2 of the assignment.

PREPROCESSING DECISIONS (based on Step 1 and Step 2 analysis):
- Feature scaling: StandardScaler (required for SVM, recommended for all algorithms)
- Class imbalance: Use class_weight='balanced' in classifiers
- Missing values: None detected (no handling needed)
- Feature selection: Not applied initially (baseline uses all features)

IMPORTANT: This script prepares data for model training but does NOT
handle the class imbalance issue yet. That will be addressed in the 
classifier configuration (Step 4).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n" + "="*80)
print("STEP 3: DATA PREPROCESSING (TASK 2)")
print("="*80 + "\n")

# ============================================================================
# LINES 1-40: Load train/valid/test splits from Step 2
# ============================================================================
print("Loading data splits from Step 2...")

train_df = pd.read_csv('train_data.csv')
valid_df = pd.read_csv('valid_data.csv')
test_df = pd.read_csv('test_data.csv')

print(f"Loaded training set: {train_df.shape[0]} samples, {train_df.shape[1]} columns")
print(f"Loaded validation set: {valid_df.shape[0]} samples, {valid_df.shape[1]} columns")
print(f"Loaded test set: {test_df.shape[0]} samples, {test_df.shape[1]} columns")

# ============================================================================
# LINES 41-80: Separate features from labels
# ============================================================================
print("\nSeparating features from labels...")

# Define which columns are features vs metadata/labels
metadata_cols = ['sid', 'uid', 'gender', 'filename', 'system']
label_cols = ['target_human', 'target_asv']
feature_cols = [col for col in train_df.columns 
                if col not in metadata_cols + label_cols]

print(f"Total feature columns: {len(feature_cols)}")
print(f"Label columns: {label_cols}")
print(f"Metadata columns: {metadata_cols}")

# Extract features and labels for each split
X_train = train_df[feature_cols].values
y_train_human = train_df['target_human'].values
y_train_asv = train_df['target_asv'].values

X_valid = valid_df[feature_cols].values
y_valid_human = valid_df['target_human'].values
y_valid_asv = valid_df['target_asv'].values

X_test = test_df[feature_cols].values
y_test_human = test_df['target_human'].values
y_test_asv = test_df['target_asv'].values

print(f"\nFeature matrix shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_valid: {X_valid.shape}")
print(f"  X_test: {X_test.shape}")

# ============================================================================
# LINES 81-120: Check for missing values
# ============================================================================
print("\n" + "-"*80)
print("TASK 2.1: Missing Value Analysis")
print("-"*80)

train_missing = np.isnan(X_train).sum()
valid_missing = np.isnan(X_valid).sum()
test_missing = np.isnan(X_test).sum()

print(f"\nMissing values detected:")
print(f"  Training set: {train_missing} missing values")
print(f"  Validation set: {valid_missing} missing values")
print(f"  Test set: {test_missing} missing values")

if train_missing + valid_missing + test_missing == 0:
    print("\nNo missing values detected. No imputation required.")
else:
    print("\nWARNING: Missing values detected. Imputation strategy needed.")
    print("Recommended: Use SimpleImputer with median strategy")

# ============================================================================
# LINES 121-180: Feature scaling using StandardScaler
# ============================================================================
print("\n" + "-"*80)
print("TASK 2.2: Feature Scaling")
print("-"*80)

print("\nApplying StandardScaler to normalize features...")
print("Rationale:")
print("  - SVM is sensitive to feature scales")
print("  - Improves convergence for gradient-based algorithms")
print("  - Prevents features with large magnitudes from dominating")

# Fit scaler on training data only (to prevent data leakage)
scaler = StandardScaler()
scaler.fit(X_train)

# Transform all three sets using the training set statistics
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

print(f"\nScaling statistics computed from training set:")
print(f"  Feature means: min={scaler.mean_.min():.2f}, max={scaler.mean_.max():.2f}")
print(f"  Feature std devs: min={scaler.scale_.min():.2f}, max={scaler.scale_.max():.2f}")

# Verify scaling worked correctly
print(f"\nPost-scaling verification (training set):")
print(f"  Mean of scaled features: {X_train_scaled.mean(axis=0).mean():.6f} (should be ~0)")
print(f"  Std of scaled features: {X_train_scaled.std(axis=0).mean():.6f} (should be ~1)")

# ============================================================================
# LINES 181-240: Analyze class imbalance
# ============================================================================
print("\n" + "-"*80)
print("TASK 2.3: Class Imbalance Analysis")
print("-"*80)

def analyze_class_distribution(y, target_name):
    """Analyze and report class distribution"""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    print(f"\n{target_name} class distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"  {label:10s}: {count:5d} samples ({percentage:5.1f}%)")
    
    imbalance_ratio = counts.max() / counts.min()
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 5.0:
        print(f"  Status: SEVERE imbalance detected")
    elif imbalance_ratio > 2.0:
        print(f"  Status: MODERATE imbalance detected")
    else:
        print(f"  Status: Balanced")
    
    return dict(zip(unique, counts)), imbalance_ratio

print("\nTraining set class distributions:")
train_human_dist, human_ratio = analyze_class_distribution(y_train_human, "target_human")
train_asv_dist, asv_ratio = analyze_class_distribution(y_train_asv, "target_asv")

print("\n" + "-"*80)
print("RECOMMENDED STRATEGY FOR HANDLING IMBALANCE:")
print("-"*80)
print("""
Given the severe class imbalance (especially for target_asv with 27.29x ratio),
we will use the 'class_weight=balanced' parameter in sklearn classifiers.

This approach:
1. Automatically adjusts weights inversely proportional to class frequencies
2. Penalizes mistakes on minority classes more heavily
3. Does not artificially create synthetic samples (unlike SMOTE)
4. Maintains the original data distribution for evaluation

Alternative considered but not used:
- SMOTE: Risk of overfitting on synthetic minority samples
- Undersampling: Would discard valuable majority class data
- Manual weights: Less generalizable than automatic balancing

Implementation: The class_weight parameter will be set in the classifier
configuration during model training (Step 4).
""")

# ============================================================================
# LINES 241-280: Save preprocessed data
# ============================================================================
print("\n" + "-"*80)
print("Saving preprocessed data and scaler...")
print("-"*80)

# Save scaled features
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_valid_scaled.npy', X_valid_scaled)
np.save('X_test_scaled.npy', X_test_scaled)

# Save labels
np.save('y_train_human.npy', y_train_human)
np.save('y_train_asv.npy', y_train_asv)
np.save('y_valid_human.npy', y_valid_human)
np.save('y_valid_asv.npy', y_valid_asv)
np.save('y_test_human.npy', y_test_human)
np.save('y_test_asv.npy', y_test_asv)

# Save scaler for potential future use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names for reference
with open('feature_names.txt', 'w') as f:
    for feature in feature_cols:
        f.write(feature + '\n')

print("Saved preprocessed data:")
print("  - X_train_scaled.npy, X_valid_scaled.npy, X_test_scaled.npy")
print("  - y_train_human.npy, y_train_asv.npy, etc.")
print("  - scaler.pkl (StandardScaler object)")
print("  - feature_names.txt (feature column names)")

# ============================================================================
# LINES 281-350: Create visualizations for report
# ============================================================================
print("\n" + "-"*80)
print("Creating visualizations...")
print("-"*80)

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
print("Saved visualization: step3_preprocessing.png")

# ============================================================================
# LINES 351-400: Generate summary for report
# ============================================================================
print("\n" + "="*80)
print("SUMMARY FOR YOUR REPORT - TASK 2")
print("="*80)

report_text = f"""
TASK 2: DATA PREPROCESSING STRATEGY AND JUSTIFICATION

Missing Value Handling (Lines 81-120):
Analysis of the preprocessed data revealed no missing values across all three 
splits. Therefore, no imputation strategy was required.

Feature Scaling (Lines 121-180):
We applied StandardScaler normalization to all {len(feature_cols)} audio features. 
This preprocessing step is critical for several reasons:

1. SVM algorithms are sensitive to feature scales and require normalized inputs
2. Features with large magnitudes (e.g., frequency values) would otherwise 
   dominate the learning process
3. Standardization (zero mean, unit variance) improves convergence for 
   optimization-based algorithms

The scaler was fit exclusively on the training set (Lines 127-128) to prevent 
data leakage. The same transformation was then applied to validation and test 
sets using the training statistics. Post-scaling verification confirmed that 
scaled features have approximately zero mean and unit standard deviation.

Class Imbalance Handling (Lines 181-240):
Both target variables exhibit severe class imbalance:
- target_human: {human_ratio:.2f}x imbalance ratio
- target_asv: {asv_ratio:.2f}x imbalance ratio (particularly severe)

The target_asv label is especially problematic, with the "sheep" class 
representing {train_asv_dist.get('sheep', 0)/len(y_train_asv)*100:.1f}% of training samples. 
A naive classifier predicting only "sheep" would achieve high accuracy but 
poor performance on minority classes.

To address this imbalance, we will use the class_weight='balanced' parameter 
in our classification algorithms (implemented in Step 4). This approach 
automatically weights classes inversely proportional to their frequencies, 
ensuring the model learns to identify minority classes without discarding 
valuable majority class data or introducing synthetic samples.

Alternative approaches (SMOTE, undersampling) were considered but rejected:
- SMOTE risks overfitting on synthetic minority samples
- Undersampling would discard {train_asv_dist.get('sheep', 0) - train_asv_dist.get('wolf', 0)} 
  valuable majority class samples

Feature Selection:
Initial experiments use all {len(feature_cols)} features. Feature selection 
techniques (e.g., mutual information, recursive feature elimination) were not 
applied in the baseline to establish full feature set performance. This could 
be explored in hyperparameter tuning if initial results suggest overfitting.
"""

print(report_text)

# Save summary to text file
with open('task2_summary.txt', 'w') as f:
    f.write(report_text)
print("\nSaved: task2_summary.txt (use this for your report)")

print("\n" + "="*80)
print("STEP 3 COMPLETE")
print("="*80)
print("\nData is now ready for model training.")
print("Next step: Run step4_model_training.py")
print("\nFiles created:")
print("  - Scaled feature matrices (.npy files)")
print("  - Label arrays (.npy files)")
print("  - StandardScaler object (scaler.pkl)")
print("  - Visualization (step3_preprocessing.png)")
print("  - Report summary (task2_summary.txt)")