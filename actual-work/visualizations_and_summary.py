"""
AICE2002 Assignment 1 - Visualizations and Summary
This file contains visualization code and summary text for the report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_data_exploration_visualizations(df_orig, df_X, df_Y, speakers, gender_counts, 
                                          samples_per_speaker, human_dist, asv_dist):
    """Create visualizations for Step 1: Data Exploration"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Exploration - Step 1', fontsize=16, fontweight='bold')
    
    # Plot 1: Class distribution for target_human
    ax1 = axes[0, 0]
    human_dist.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Class Distribution: target_human', fontweight='bold')
    ax1.set_xlabel('Class Label')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(human_dist.values):
        ax1.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Class distribution for target_asv
    ax2 = axes[0, 1]
    asv_dist.plot(kind='bar', ax=ax2, color='lightcoral', edgecolor='black')
    ax2.set_title('Class Distribution: target_asv', fontweight='bold')
    ax2.set_xlabel('Class Label')
    ax2.set_ylabel('Count')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(asv_dist.values):
        ax2.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Samples per speaker
    ax3 = axes[1, 0]
    samples_per_speaker.plot(kind='hist', bins=20, ax=ax3, color='lightgreen', edgecolor='black')
    ax3.set_title('Distribution of Samples per Speaker', fontweight='bold')
    ax3.set_xlabel('Number of Samples')
    ax3.set_ylabel('Frequency')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Gender distribution
    ax4 = axes[1, 1]
    gender_counts.plot(kind='bar', ax=ax4, color=['steelblue', 'pink'], edgecolor='black')
    ax4.set_title('Speaker Gender Distribution', fontweight='bold')
    ax4.set_xlabel('Gender')
    ax4.set_ylabel('Number of Speakers')
    ax4.set_xticklabels(['Female', 'Male'], rotation=0)
    ax4.grid(axis='y', alpha=0.3)
    for i, v in enumerate(gender_counts.values):
        ax4.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('step1_data_exploration.png', dpi=300, bbox_inches='tight')
    print("Saved: step1_data_exploration.png")


def create_data_splitting_visualizations(train_df, valid_df, test_df, train_speakers, 
                                       valid_speakers, test_speakers, train_human, 
                                       valid_human, test_human, train_asv, valid_asv, test_asv):
    """Create visualizations for Step 2: Data Splitting"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Data Split Analysis - Step 2', fontsize=16, fontweight='bold')
    
    # Plot 1: Sample distribution across splits
    ax1 = axes[0, 0]
    split_sizes = [len(train_df), len(valid_df), len(test_df)]
    split_labels = [f'Train\n{len(train_df)}', f'Valid\n{len(valid_df)}', f'Test\n{len(test_df)}']
    ax1.bar(range(3), split_sizes, color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(split_labels)
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Sample Distribution Across Splits', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: target_human in train set
    ax2 = axes[0, 1]
    train_human.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
    ax2.set_title('Train Set - target_human', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: target_asv in train set
    ax3 = axes[0, 2]
    train_asv.plot(kind='bar', ax=ax3, color='lightcoral', edgecolor='black')
    ax3.set_title('Train Set - target_asv', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Count')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: target_human across all splits
    ax4 = axes[1, 0]
    all_classes_human = sorted(set(train_human.index) | set(valid_human.index) | set(test_human.index))
    x_human = np.arange(len(all_classes_human))
    width = 0.25
    
    train_human_aligned = [train_human.get(c, 0) for c in all_classes_human]
    valid_human_aligned = [valid_human.get(c, 0) for c in all_classes_human]
    test_human_aligned = [test_human.get(c, 0) for c in all_classes_human]
    
    ax4.bar(x_human - width, train_human_aligned, width, label='Train', color='#2ecc71', edgecolor='black')
    ax4.bar(x_human, valid_human_aligned, width, label='Valid', color='#3498db', edgecolor='black')
    ax4.bar(x_human + width, test_human_aligned, width, label='Test', color='#e74c3c', edgecolor='black')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Count')
    ax4.set_title('target_human Distribution Across Splits', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_human)
    ax4.set_xticklabels(all_classes_human, rotation=45)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: target_asv across all splits
    ax5 = axes[1, 1]
    all_classes_asv = sorted(set(train_asv.index) | set(valid_asv.index) | set(test_asv.index))
    x_asv = np.arange(len(all_classes_asv))
    
    train_asv_aligned = [train_asv.get(c, 0) for c in all_classes_asv]
    valid_asv_aligned = [valid_asv.get(c, 0) for c in all_classes_asv]
    test_asv_aligned = [test_asv.get(c, 0) for c in all_classes_asv]
    
    ax5.bar(x_asv - width, train_asv_aligned, width, label='Train', color='#2ecc71', edgecolor='black')
    ax5.bar(x_asv, valid_asv_aligned, width, label='Valid', color='#3498db', edgecolor='black')
    ax5.bar(x_asv + width, test_asv_aligned, width, label='Test', color='#e74c3c', edgecolor='black')
    ax5.set_xlabel('Class')
    ax5.set_ylabel('Count')
    ax5.set_title('target_asv Distribution Across Splits', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_asv)
    ax5.set_xticklabels(all_classes_asv, rotation=45)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Speaker distribution
    ax6 = axes[1, 2]
    speaker_counts = [len(train_speakers), len(valid_speakers), len(test_speakers)]
    ax6.bar(range(3), speaker_counts, color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
    ax6.set_xticks(range(3))
    ax6.set_xticklabels([f'Train\n{len(train_speakers)}', f'Valid\n{len(valid_speakers)}', f'Test\n{len(test_speakers)}'])
    ax6.set_ylabel('Number of Speakers', fontsize=12, fontweight='bold')
    ax6.set_title('Speaker Distribution Across Splits', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('step2_data_splitting.png', dpi=300, bbox_inches='tight')
    print("Saved: step2_data_splitting.png")


def create_preprocessing_visualizations(X_train, X_train_scaled, train_human_dist, 
                                       train_asv_dist, human_ratio, asv_ratio):
    """Create visualizations for Step 3: Data Preprocessing"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Data Preprocessing Analysis - Step 3', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature distribution before scaling
    ax1 = axes[0, 0]
    sample_features = np.random.choice(X_train.shape[1], 5, replace=False)
    for i in sample_features:
        ax1.hist(X_train[:, i], bins=30, alpha=0.5, label=f'Feature {i}')
    ax1.set_xlabel('Feature Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Feature Distributions Before Scaling', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Feature distribution after scaling
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
    feature_means_after = X_train_scaled.mean(axis=0)
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
    ax4.bar(range(len(labels_human)), counts_human, edgecolor='black')
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
    ax5.bar(range(len(labels_asv)), counts_asv, edgecolor='black', color='coral')
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
    ax6.bar(range(2), ratios, color=colors, edgecolor='black')
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
    print("Saved: step3_preprocessing.png")


def create_confusion_matrices(results, filename='confusion_matrices.png'):
    """Create confusion matrix visualizations for model results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    plot_configs = [
        ('Random Forest_human', axes[0, 0]),
        ('Random Forest_asv', axes[0, 1]),
        ('SVM_human', axes[1, 0]),
        ('SVM_asv', axes[1, 1])
    ]
    
    for key, ax in plot_configs:
        if key in results:
            result = results[key]
            model_name, target = key.rsplit('_', 1)
            
            sns.heatmap(result['confusion_matrix'], 
                        annot=True, fmt='d', cmap='Blues',
                        xticklabels=result['classes'],
                        yticklabels=result['classes'],
                        ax=ax, cbar_kws={'label': 'Count'})
            
            ax.set_xlabel('Predicted Label', fontweight='bold')
            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_title(f"{model_name} - target_{target}\n"
                        f"Acc: {result['test_accuracy']:.3f}, F1: {result['test_f1_macro']:.3f}",
                        fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def create_hyperparameter_tuning_visualizations(improvements):
    """Create visualizations for Step 5: Hyperparameter Tuning"""
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
    print("Saved: hyperparameter_tuning_results.png")


def generate_summary_text():
    """Generate summary text for the report"""
    summary = """
TASK 1: DATA SPLITTING STRATEGY

The data splitting was performed using speaker-based stratified splitting to prevent data leakage.
All three systems (system_Original, system_X, system_Y) were combined into a single dataset.
Speakers were split into train/valid/test sets with approximately 70/15/15 distribution.
Each speaker appears in only one split to ensure the model generalizes to new speakers.

TASK 2: DATA PREPROCESSING

Missing values were checked and none were found across all splits.
Feature scaling was applied using StandardScaler, fit on the training set only.
Both target variables exhibit severe class imbalance, which was addressed using
class weighting and SMOTE oversampling in the model training phase.

TASK 3: ALGORITHM SELECTION

Two algorithms were selected:
1. Random Forest Classifier - handles high-dimensional features effectively
2. Support Vector Machine (SVM) with RBF kernel - effective for non-linear classification

Both algorithms were configured with class_weight='balanced' and SMOTE oversampling
to address the severe class imbalance.

TASK 5: EVALUATION METRICS

Evaluation was performed using:
- Accuracy (overall performance)
- F1-macro (important for imbalanced data)
- F1-micro (alternative averaging)
- Per-class precision, recall, and F1 scores
- Confusion matrices for each model-target combination

TASK 6: HYPERPARAMETER TUNING

GridSearchCV was used with 3-fold cross-validation to optimize hyperparameters.
F1-macro was used as the scoring metric due to class imbalance.
Search spaces included:
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf
- SVM: C, gamma

Results showed improvements in F1-macro scores after hyperparameter tuning.
"""
    return summary


if __name__ == "__main__":
    print("This file contains visualization and summary functions.")
    print("Import and use the functions as needed for your report.")

