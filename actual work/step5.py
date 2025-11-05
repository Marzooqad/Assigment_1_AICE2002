"""
AICE2002 Assignment 1 - STEP 5: Hyperparameter Tuning (TASK 6)
Author: [Your Name]
Date: November 2, 2025

PURPOSE: Optimize model hyperparameters using GridSearchCV to improve 
performance. This addresses Task 6 of the assignment.

HYPERPARAMETER SEARCH SPACES:
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf
- SVM: C, gamma

VALIDATION STRATEGY:
- Use validation set for hyperparameter selection
- Report improvements over baseline on test set
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import time

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n" + "="*80)
print("STEP 5: HYPERPARAMETER TUNING (TASK 6)")
print("="*80 + "\n")

# ============================================================================
# LINES 1-40: Load preprocessed data
# ============================================================================
print("Loading preprocessed data...")

X_train = np.load('X_train_scaled.npy')
X_valid = np.load('X_valid_scaled.npy')
X_test = np.load('X_test_scaled.npy')

y_train_human = np.load('y_train_human.npy', allow_pickle=True)
y_train_asv = np.load('y_train_asv.npy', allow_pickle=True)
y_valid_human = np.load('y_valid_human.npy', allow_pickle=True)
y_valid_asv = np.load('y_valid_asv.npy', allow_pickle=True)
y_test_human = np.load('y_test_human.npy', allow_pickle=True)
y_test_asv = np.load('y_test_asv.npy', allow_pickle=True)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_valid.shape}")
print(f"Test set: {X_test.shape}")

# Combine train and validation for GridSearchCV
X_train_full = np.vstack([X_train, X_valid])
y_train_full_human = np.concatenate([y_train_human, y_valid_human])
y_train_full_asv = np.concatenate([y_train_asv, y_valid_asv])

print(f"Combined train+valid: {X_train_full.shape}")

# ============================================================================
# LINES 41-100: Define hyperparameter search spaces
# ============================================================================
print("\n" + "-"*80)
print("TASK 6: HYPERPARAMETER SEARCH SPACES")
print("-"*80)

print("""
Hyperparameter Search Strategy:

1. Random Forest Parameters:
   - n_estimators: [50, 100, 200] - Number of trees in forest
     Rationale: More trees generally improve performance but increase computation
   
   - max_depth: [10, 20, 30, None] - Maximum tree depth
     Rationale: Controls overfitting. None allows trees to expand until pure
   
   - min_samples_split: [2, 5, 10] - Minimum samples to split internal node
     Rationale: Higher values prevent overfitting on small groups
   
   - min_samples_leaf: [1, 2, 4] - Minimum samples in leaf node
     Rationale: Regularization to prevent overfitting

2. SVM Parameters:
   - C: [0.1, 1.0, 10.0] - Regularization parameter
     Rationale: Lower C = more regularization (simpler decision boundary)
   
   - gamma: [0.001, 0.01, 0.1, 'scale'] - Kernel coefficient
     Rationale: Controls influence of single training examples
     'scale' = 1 / (n_features * X.var())

Validation Strategy:
- 3-fold cross-validation on combined train+valid set
- Scoring metric: F1-macro (appropriate for imbalanced data)
- Best parameters selected based on cross-validation F1-macro
""")

# Define parameter grids
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

svm_param_grid = {
    'C': [0.1, 1.0, 10.0],
    'gamma': [0.001, 0.01, 0.1, 'scale']
}

print(f"\nRandom Forest search space: {len(rf_param_grid['n_estimators']) * len(rf_param_grid['max_depth']) * len(rf_param_grid['min_samples_split']) * len(rf_param_grid['min_samples_leaf'])} combinations")
print(f"SVM search space: {len(svm_param_grid['C']) * len(svm_param_grid['gamma'])} combinations")

# ============================================================================
# LINES 101-250: Hyperparameter tuning with GridSearchCV
# ============================================================================
print("\n" + "-"*80)
print("Running GridSearchCV (this will take several minutes)...")
print("-"*80)

# F1-macro scorer for imbalanced data
f1_scorer = make_scorer(f1_score, average='macro')

results_tuning = {}

def tune_and_evaluate(model, param_grid, model_name, X_train, y_train, X_test, y_test, target_name):
    """
    Perform hyperparameter tuning and evaluate on test set.
    """
    print(f"\n  Tuning {model_name} on {target_name}...")
    
    # GridSearchCV with 3-fold cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=0
    )
    
    # Fit grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tune_time = time.time() - start_time
    
    print(f"    Tuning completed in {tune_time:.1f} seconds")
    print(f"    Best CV F1-macro: {grid_search.best_score_:.4f}")
    print(f"    Best parameters: {grid_search.best_params_}")
    
    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"    Test set - Accuracy: {test_accuracy:.4f}, F1-macro: {test_f1_macro:.4f}")
    
    return {
        'model_name': model_name,
        'target': target_name,
        'best_params': grid_search.best_params_,
        'best_cv_f1': grid_search.best_score_,
        'test_accuracy': test_accuracy,
        'test_f1_macro': test_f1_macro,
        'tune_time': tune_time,
        'cv_results': grid_search.cv_results_
    }

# Tune Random Forest on target_human
print("\nRandom Forest:")
rf_human = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1
)
results_tuning['RF_human'] = tune_and_evaluate(
    rf_human, rf_param_grid, 'Random Forest',
    X_train_full, y_train_full_human,
    X_test, y_test_human,
    'target_human'
)

# Tune Random Forest on target_asv
rf_asv = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1
)
results_tuning['RF_asv'] = tune_and_evaluate(
    rf_asv, rf_param_grid, 'Random Forest',
    X_train_full, y_train_full_asv,
    X_test, y_test_asv,
    'target_asv'
)

# Tune SVM on target_human
print("\nSVM:")
svm_human = SVC(
    kernel='rbf',
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
results_tuning['SVM_human'] = tune_and_evaluate(
    svm_human, svm_param_grid, 'SVM',
    X_train_full, y_train_full_human,
    X_test, y_test_human,
    'target_human'
)

# Tune SVM on target_asv
svm_asv = SVC(
    kernel='rbf',
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
results_tuning['SVM_asv'] = tune_and_evaluate(
    svm_asv, svm_param_grid, 'SVM',
    X_train_full, y_train_full_asv,
    X_test, y_test_asv,
    'target_asv'
)

# ============================================================================
# LINES 251-300: Compare baseline vs tuned performance
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE COMPARISON: BASELINE vs TUNED")
print("="*80)

# Load baseline results from Step 4
baseline_results = pd.read_csv('model_results.csv')

print("\nBaseline Performance (from Step 4):")
print(baseline_results[['Model', 'Target', 'Test_Accuracy', 'Test_F1_Macro']].to_string(index=False))

print("\n" + "-"*80)
print("Tuned Performance:")
print("-"*80)
print(f"{'Model':<20} {'Target':<15} {'Accuracy':<12} {'F1-macro':<12}")
print("-"*80)
for key, result in results_tuning.items():
    print(f"{result['model_name']:<20} {result['target']:<15} "
          f"{result['test_accuracy']:.4f}       {result['test_f1_macro']:.4f}")
print("-"*80)

# Calculate improvements
print("\n" + "-"*80)
print("Performance Improvements:")
print("-"*80)

improvements = []
for key, tuned in results_tuning.items():
    # Find corresponding baseline result
    model_name = tuned['model_name']
    target = tuned['target']
    
    baseline_row = baseline_results[
        (baseline_results['Model'] == model_name) & 
        (baseline_results['Target'] == target)
    ]
    
    if not baseline_row.empty:
        baseline_f1 = baseline_row['Test_F1_Macro'].values[0]
        tuned_f1 = tuned['test_f1_macro']
        improvement = tuned_f1 - baseline_f1
        pct_improvement = (improvement / baseline_f1 * 100) if baseline_f1 > 0 else 0
        
        print(f"{model_name} - {target}:")
        print(f"  Baseline F1-macro: {baseline_f1:.4f}")
        print(f"  Tuned F1-macro:    {tuned_f1:.4f}")
        print(f"  Improvement:       {improvement:+.4f} ({pct_improvement:+.1f}%)\n")
        
        improvements.append({
            'Model': model_name,
            'Target': target,
            'Baseline_F1': baseline_f1,
            'Tuned_F1': tuned_f1,
            'Improvement': improvement,
            'Pct_Improvement': pct_improvement
        })

improvements_df = pd.DataFrame(improvements)
improvements_df.to_csv('hyperparameter_improvements.csv', index=False)
print("Saved: hyperparameter_improvements.csv")

# ============================================================================
# LINES 301-350: Create best parameters table
# ============================================================================
print("\n" + "-"*80)
print("Best Hyperparameters Found:")
print("-"*80)

best_params_list = []
for key, result in results_tuning.items():
    best_params_list.append({
        'Model': result['model_name'],
        'Target': result['target'],
        **result['best_params'],
        'CV_F1_Macro': result['best_cv_f1']
    })

best_params_df = pd.DataFrame(best_params_list)
print(best_params_df.to_string(index=False))

best_params_df.to_csv('best_hyperparameters.csv', index=False)
print("\nSaved: best_hyperparameters.csv")

# ============================================================================
# LINES 351-420: Create visualizations
# ============================================================================
print("\n" + "-"*80)
print("Creating visualizations...")
print("-"*80)

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
