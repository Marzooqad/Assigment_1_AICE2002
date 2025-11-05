"""
AICE2002 Assignment 1 - STEP 4 IMPROVED: Combined SMOTE + Manual Weights (TASKS 3, 5, 6)
Author: [Your Name]
Date: November 2, 2025

PURPOSE: Train classification models with SMOTE oversampling AND aggressive manual
class weights to address severe class imbalance and improve minority class detection.

IMPROVEMENTS FROM BASELINE:
- Added SMOTE (Synthetic Minority Over-sampling Technique) to balance training data
- Added aggressive manual class weights (30-50x penalty for minority classes)
- Combined approach addresses both data scarcity and learning bias
- This addresses the critical issue where models achieved 0% recall on minority classes

ALGORITHM SELECTION (Task 3):
- Random Forest: Handles high-dimensional audio features well
- SVM with RBF kernel: Effective for non-linear classification

EVALUATION METRICS (Task 5):
- Accuracy, F1-macro, F1-micro
- Per-class Precision, Recall, F1
- Confusion matrices (required by assignment)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n" + "="*80)
print("STEP 4 IMPROVED: MODEL TRAINING WITH SMOTE + AGGRESSIVE WEIGHTS")
print("="*80 + "\n")

# ============================================================================
# LINES 1-40: Load preprocessed data from Step 3
# ============================================================================
print("Loading preprocessed data from Step 3...")

X_train = np.load('X_train_scaled.npy')
X_valid = np.load('X_valid_scaled.npy')
X_test = np.load('X_test_scaled.npy')

y_train_human = np.load('y_train_human.npy', allow_pickle=True)
y_train_asv = np.load('y_train_asv.npy', allow_pickle=True)
y_valid_human = np.load('y_valid_human.npy', allow_pickle=True)
y_valid_asv = np.load('y_valid_asv.npy', allow_pickle=True)
y_test_human = np.load('y_test_human.npy', allow_pickle=True)
y_test_asv = np.load('y_test_asv.npy', allow_pickle=True)

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Validation set: {X_valid.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# LINES 41-100: Define classifiers (Task 3)
# ============================================================================
print("\n" + "-"*80)
print("TASK 3: ALGORITHM SELECTION AND JUSTIFICATION")
print("-"*80)

print("""
Selected Algorithms:

1. Random Forest Classifier
   Justification:
   - Handles high-dimensional feature spaces effectively (88 features)
   - Robust to outliers and does not require feature scaling
   - Provides feature importance rankings for interpretability
   - Less prone to overfitting than single decision trees
   - Commonly used for audio classification tasks
   
   Configuration:
   - n_estimators=100, max_depth=20
   - SMOTE oversampling (Lines 110-120) + Aggressive manual class weights
   - Manual weights: lamb=30-50x, wolf=15-50x higher than sheep
   
2. Support Vector Machine (SVM) with RBF Kernel
   Justification:
   - Effective for non-linear classification problems
   - RBF kernel can capture complex decision boundaries
   - Well-established for audio feature classification
   - Requires scaled features (completed in Step 3)
   
   Configuration:
   - kernel='rbf', C=1.0, gamma='scale'
   - SMOTE oversampling (Lines 110-120) + Aggressive manual class weights
   - Manual weights: lamb=30-50x, wolf=15-50x higher than sheep
""")

# Initialize classifiers with AGGRESSIVE manual class weights
# This is more effective than SMOTE for extreme imbalance

print("\nUsing aggressive manual class weights for minority class emphasis:")
print("  target_human: sheep=1, goat=8, lamb=30, wolf=15")
print("  target_asv: sheep=1, goat=10, lamb=50, wolf=50")

classifiers_human = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        class_weight={'sheep': 1, 'goat': 8, 'lamb': 30, 'wolf': 15},
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=RANDOM_STATE,
        class_weight={'sheep': 1, 'goat': 8, 'lamb': 30, 'wolf': 15}
    )
}

classifiers_asv = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        class_weight={'sheep': 1, 'goat': 10, 'lamb': 50, 'wolf': 50},
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=RANDOM_STATE,
        class_weight={'sheep': 1, 'goat': 10, 'lamb': 50, 'wolf': 50}
    )
}

print("\nClassifiers initialized with aggressive manual class weights.")

# ============================================================================
# LINES 101-220: Train and evaluate models WITH SMOTE
# ============================================================================
print("\n" + "-"*80)
print("Training models with SMOTE oversampling on both target variables...")
print("-"*80)

results = {}

def train_and_evaluate(clf, clf_name, X_train, y_train, X_valid, y_valid, X_test, y_test, target_name):
    """
    Train a classifier with SMOTE oversampling and evaluate on validation and test sets.
    
    CRITICAL IMPROVEMENT (Lines 110-120):
    SMOTE is applied to balance the training data before model training. This
    addresses the severe class imbalance that caused 0% recall on minority classes.
    
    Args:
        clf: Classifier object
        clf_name: Name of classifier (for reporting)
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
        target_name: Name of target variable
    
    Returns:
        Dictionary containing trained model and evaluation metrics
    """
    print(f"\n  Training {clf_name} on {target_name}...")
    
    # Display original class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"    Original training distribution:")
    for label, count in zip(unique, counts):
        print(f"      {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # LINES 110-120: Apply SMOTE to balance training data
    # This is the KEY IMPROVEMENT over baseline
    print(f"    Applying SMOTE oversampling...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Display resampled class distribution
    unique_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
    print(f"    After SMOTE:")
    for label, count in zip(unique_resampled, counts_resampled):
        print(f"      {label}: {count} samples ({count/len(y_train_resampled)*100:.1f}%)")
    print(f"    Total samples increased: {len(y_train)} -> {len(y_train_resampled)}")
    
    # Train model on resampled data
    start_time = time.time()
    clf.fit(X_train_resampled, y_train_resampled)
    train_time = time.time() - start_time
    
    print(f"    Training completed in {train_time:.2f} seconds")
    
    # Predictions on validation set
    y_valid_pred = clf.predict(X_valid)
    
    # Predictions on test set
    y_test_pred = clf.predict(X_test)
    
    # Calculate metrics on validation set
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    valid_f1_macro = f1_score(y_valid, y_valid_pred, average='macro')
    valid_f1_micro = f1_score(y_valid, y_valid_pred, average='micro')
    
    # Calculate metrics on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1_micro = f1_score(y_test, y_test_pred, average='micro')
    
    # Per-class metrics on test set
    test_precision = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    
    # Confusion matrix on test set
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Get class labels
    classes = np.unique(np.concatenate([y_train, y_valid, y_test]))
    
    print(f"    Validation - Accuracy: {valid_accuracy:.4f}, F1-macro: {valid_f1_macro:.4f}")
    print(f"    Test       - Accuracy: {test_accuracy:.4f}, F1-macro: {test_f1_macro:.4f}")
    
    return {
        'model': clf,
        'train_time': train_time,
        'valid_accuracy': valid_accuracy,
        'valid_f1_macro': valid_f1_macro,
        'valid_f1_micro': valid_f1_micro,
        'test_accuracy': test_accuracy,
        'test_f1_macro': test_f1_macro,
        'test_f1_micro': test_f1_micro,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm,
        'classes': classes,
        'y_test_true': y_test,
        'y_test_pred': y_test_pred,
        'original_train_size': len(y_train),
        'resampled_train_size': len(y_train_resampled)
    }

# Train all models on both targets
print("\nTraining Random Forest:")

# Train on target_human
results['Random Forest_human'] = train_and_evaluate(
    classifiers_human['Random Forest'], 'Random Forest', 
    X_train, y_train_human, 
    X_valid, y_valid_human, X_test, y_test_human, 
    'target_human'
)

# Train on target_asv
results['Random Forest_asv'] = train_and_evaluate(
    classifiers_asv['Random Forest'], 'Random Forest',
    X_train, y_train_asv,
    X_valid, y_valid_asv, X_test, y_test_asv,
    'target_asv'
)

print("\nTraining SVM:")

# Train on target_human
results['SVM_human'] = train_and_evaluate(
    classifiers_human['SVM'], 'SVM',
    X_train, y_train_human,
    X_valid, y_valid_human, X_test, y_test_human,
    'target_human'
)

# Train on target_asv
results['SVM_asv'] = train_and_evaluate(
    classifiers_asv['SVM'], 'SVM',
    X_train, y_train_asv,
    X_valid, y_valid_asv, X_test, y_test_asv,
    'target_asv'
)

# ============================================================================
# LINES 221-280: Compare with baseline results
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: BASELINE vs SMOTE-IMPROVED")
print("="*80)

# Load baseline results from previous run
try:
    baseline_results = pd.read_csv('model_results.csv')
    print("\nBaseline Performance (without SMOTE):")
    print(baseline_results[['Model', 'Target', 'Test_Accuracy', 'Test_F1_Macro']].to_string(index=False))
except:
    print("\nBaseline results not found. Run original step4 first for comparison.")

print("\n" + "-"*80)
print("SMOTE-Improved Performance:")
print("-"*80)
print(f"{'Model':<20} {'Target':<15} {'Accuracy':<12} {'F1-macro':<12} {'F1-micro':<12}")
print("-"*80)

for key, result in results.items():
    model_name, target = key.rsplit('_', 1)
    print(f"{model_name:<20} {f'target_{target}':<15} "
          f"{result['test_accuracy']:.4f}       "
          f"{result['test_f1_macro']:.4f}       "
          f"{result['test_f1_micro']:.4f}")

print("-"*80)

# Save improved results
results_df = pd.DataFrame({
    'Model': [key.rsplit('_', 1)[0] for key in results.keys()],
    'Target': [f"target_{key.rsplit('_', 1)[1]}" for key in results.keys()],
    'Test_Accuracy': [result['test_accuracy'] for result in results.values()],
    'Test_F1_Macro': [result['test_f1_macro'] for result in results.values()],
    'Test_F1_Micro': [result['test_f1_micro'] for result in results.values()],
    'Valid_Accuracy': [result['valid_accuracy'] for result in results.values()],
    'Valid_F1_Macro': [result['valid_f1_macro'] for result in results.values()],
    'Method': ['SMOTE'] * len(results)
})
results_df.to_csv('model_results_smote.csv', index=False)
print("\nSaved: model_results_smote.csv")

# ============================================================================
# LINES 281-350: Detailed per-class metrics
# ============================================================================
print("\n" + "-"*80)
print("Per-Class Performance Metrics (SMOTE-Improved):")
print("-"*80)

for key, result in results.items():
    model_name, target = key.rsplit('_', 1)
    print(f"\n{model_name} - target_{target}:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*50)
    
    for i, cls in enumerate(result['classes']):
        if i < len(result['test_precision']):
            recall_val = result['test_recall'][i]
            # Highlight improvement in minority class recall
            if recall_val > 0.0:
                marker = " <-- IMPROVED!" if cls in ['lamb', 'wolf'] else ""
            else:
                marker = " (still 0%)" if cls in ['lamb', 'wolf'] else ""
            
            print(f"{cls:<10} {result['test_precision'][i]:.4f}       "
                  f"{result['test_recall'][i]:.4f}       "
                  f"{result['test_f1'][i]:.4f}{marker}")
        else:
            print(f"{cls:<10} N/A (not in test set)")
