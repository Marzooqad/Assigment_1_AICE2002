"""
AICE2002 Assignment 1 - STEP 4 FINAL: XGBoost + SVM with Extreme Weighting
Author: [Your Name]
Date: November 2, 2025

PURPOSE: Final attempt to detect lamb class using XGBoost (better than Random Forest
for imbalanced data) with EXTREME class weights.

ALGORITHM CHANGES:
- REMOVED: Random Forest
- ADDED: XGBoost (Gradient Boosting) - superior for imbalanced classification
- KEPT: SVM (already showing some wolf detection)
- INCREASED: Class weights to EXTREME levels (lamb=100x)

JUSTIFICATION FOR XGBOOST:
- Gradient boosting learns from previous errors iteratively
- Better at handling minority classes than Random Forest
- Built-in support for class imbalance via scale_pos_weight
- Standard in Kaggle competitions for imbalanced data
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                                recall_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n" + "="*80)
print("STEP 4 FINAL: XGBOOST + SVM WITH EXTREME WEIGHTING")
print("="*80 + "\n")

# ============================================================================
# Load preprocessed data
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
# Define classifiers with EXTREME class weights
# ============================================================================
print("\n" + "-"*80)
print("ALGORITHM SELECTION - FINAL APPROACH")
print("-"*80)

print("""
Selected Algorithms:

1. XGBoost (Extreme Gradient Boosting)
   Justification:
   - Superior to Random Forest for imbalanced classification
   - Learns iteratively from errors, focusing on hard-to-classify samples
   - Built-in regularization prevents overfitting on synthetic SMOTE samples
   - Industry standard for imbalanced data (fraud detection, anomaly detection)
   
   Configuration:
   - n_estimators=200 (more trees for minority class learning)
   - max_depth=6 (moderate depth to prevent overfitting)
   - learning_rate=0.1 (standard)
   - EXTREME manual class weights: lamb=100x, wolf=50x

2. Support Vector Machine (SVM) with RBF Kernel
   Justification:
   - Already showing wolf detection capability (31% recall in previous run)
   - RBF kernel can capture complex non-linear boundaries
   - Effective for high-dimensional audio features
   
   Configuration:
   - kernel='rbf', C=10.0 (increased for more complex boundaries)
   - gamma='scale'
   - EXTREME manual class weights: lamb=100x, wolf=50x
""")

# EXTREME class weights - last attempt to force lamb detection
print("\nUsing EXTREME manual class weights:")
print("  target_human: sheep=1, goat=10, lamb=100, wolf=50")
print("  target_asv: sheep=1, goat=15, lamb=100, wolf=50")
print("\nInterpretation: Model treats misclassifying 1 lamb as bad as")
print("                 misclassifying 100 sheep samples!")

# Convert class labels to numeric for XGBoost, then map weights
def get_sample_weights(y, class_weight_dict):
    """Convert class weight dictionary to sample weights"""
    weights = np.array([class_weight_dict.get(label, 1.0) for label in y])
    return weights

class_weights_human = {'sheep': 1, 'goat': 10, 'lamb': 100, 'wolf': 50}
class_weights_asv = {'sheep': 1, 'goat': 15, 'lamb': 100, 'wolf': 50}

classifiers_human = {
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    'SVM': SVC(
        kernel='rbf',
        C=10.0,  # Increased from 1.0 for more complex boundaries
        gamma='scale',
        random_state=RANDOM_STATE,
        class_weight=class_weights_human
    )
}

classifiers_asv = {
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    'SVM': SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        random_state=RANDOM_STATE,
        class_weight=class_weights_asv
    )
}

print("\nClassifiers initialized.")

# ============================================================================
# Training function with SMOTE + sample weights
# ============================================================================

results = {}

def train_and_evaluate(clf, clf_name, X_train, y_train, X_valid, y_valid, 
                        X_test, y_test, target_name, class_weights):
    """Train classifier with SMOTE and sample weights"""
    
    print(f"\n  Training {clf_name} on {target_name}...")
    
    # Display original distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"    Original training distribution:")
    for label, count in zip(unique, counts):
        print(f"      {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Apply SMOTE
    print(f"    Applying SMOTE oversampling...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Display resampled distribution
    unique_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
    print(f"    After SMOTE:")
    for label, count in zip(unique_resampled, counts_resampled):
        print(f"      {label}: {count} samples ({count/len(y_train_resampled)*100:.1f}%)")
    
    # For XGBoost: encode labels to integers
    if clf_name == 'XGBoost':
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_resampled)
        y_valid_encoded = label_encoder.transform(y_valid)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Create sample weights for encoded labels
        # Map original class weights to encoded integers
        class_to_encoded = {label: label_encoder.transform([label])[0] 
                            for label in class_weights.keys()}
        encoded_weights = {class_to_encoded[k]: v for k, v in class_weights.items()}
        sample_weights = np.array([encoded_weights.get(label, 1.0) 
                                    for label in y_train_encoded])
    else:
        # SVM uses original string labels
        y_train_encoded = y_train_resampled
        y_valid_encoded = y_valid
        y_test_encoded = y_test
        sample_weights = get_sample_weights(y_train_resampled, class_weights)
    
    # Train model
    start_time = time.time()
    
    if clf_name == 'XGBoost':
        clf.fit(X_train_resampled, y_train_encoded, sample_weight=sample_weights, verbose=False)
        # Predict with encoded labels, then decode
        y_valid_pred_encoded = clf.predict(X_valid)
        y_test_pred_encoded = clf.predict(X_test)
        y_valid_pred = label_encoder.inverse_transform(y_valid_pred_encoded)
        y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)
    else:
        # SVM already has class_weight in initialization
        clf.fit(X_train_resampled, y_train_encoded)
        y_valid_pred = clf.predict(X_valid)
        y_test_pred = clf.predict(X_test)
    
    train_time = time.time() - start_time
    print(f"    Training completed in {train_time:.2f} seconds")
    
    # Metrics (use original string labels)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1_micro = f1_score(y_test, y_test_pred, average='micro')
    
    test_precision = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_test, y_test_pred)
    classes = np.unique(np.concatenate([y_train, y_valid, y_test]))
    
    print(f"    Test - Accuracy: {test_accuracy:.4f}, F1-macro: {test_f1_macro:.4f}")
    
    return {
        'model': clf,
        'train_time': train_time,
        'test_accuracy': test_accuracy,
        'test_f1_macro': test_f1_macro,
        'test_f1_micro': test_f1_micro,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm,
        'classes': classes,
        'y_test_true': y_test,
        'y_test_pred': y_test_pred
    }

# ============================================================================
# Train all models
# ============================================================================
print("\n" + "-"*80)
print("Training models with SMOTE + EXTREME weights...")
print("-"*80)

print("\nTraining on target_human:")
for clf_name in ['XGBoost', 'SVM']:
    results[f'{clf_name}_human'] = train_and_evaluate(
        classifiers_human[clf_name], clf_name,
        X_train, y_train_human,
        X_valid, y_valid_human,
        X_test, y_test_human,
        'target_human',
        class_weights_human
    )

print("\nTraining on target_asv:")
for clf_name in ['XGBoost', 'SVM']:
    results[f'{clf_name}_asv'] = train_and_evaluate(
        classifiers_asv[clf_name], clf_name,
        X_train, y_train_asv,
        X_valid, y_valid_asv,
        X_test, y_test_asv,
        'target_asv',
        class_weights_asv
    )

# ============================================================================
# Results summary
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS - XGBOOST + SVM")
print("="*80)

print(f"\n{'Model':<15} {'Target':<15} {'Accuracy':<12} {'F1-macro':<12}")
print("-"*60)
for key, result in results.items():
    model_name, target = key.rsplit('_', 1)
    print(f"{model_name:<15} {f'target_{target}':<15} "
            f"{result['test_accuracy']:.4f}       {result['test_f1_macro']:.4f}")

# ============================================================================
# Per-class metrics - CRITICAL CHECK FOR LAMB
# ============================================================================
print("\n" + "-"*80)
print("PER-CLASS METRICS (Focus on LAMB detection):")
print("-"*80)

lamb_detected = False

for key, result in results.items():
    model_name, target = key.rsplit('_', 1)
    print(f"\n{model_name} - target_{target}:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*50)
    
    for i, cls in enumerate(result['classes']):
        if i < len(result['test_precision']):
            recall_val = result['test_recall'][i]
            
            # Check if we detected ANY lamb
            if cls == 'lamb' and recall_val > 0.0:
                lamb_detected = True
                marker = " *** LAMB DETECTED! ***"
            elif cls in ['lamb', 'wolf'] and recall_val > 0.0:
                marker = " <-- IMPROVED!"
            elif cls in ['lamb', 'wolf']:
                marker = " (still 0%)"
            else:
                marker = ""
            
            print(f"{cls:<10} {result['test_precision'][i]:.4f}       "
                    f"{result['test_recall'][i]:.4f}       "
                    f"{result['test_f1'][i]:.4f}{marker}")

if lamb_detected:
    print("\n" + "="*80)
    print("SUCCESS: At least one lamb sample was detected!")
    print("="*80)
else:
    print("\n" + "="*80)
    print("LIMITATION: Lamb class remains undetectable despite extreme measures.")
    print("This suggests lamb and sheep are inherently difficult to distinguish")
    print("in the given 88-dimensional audio feature space.")
    print("="*80)

# ============================================================================
# Save results
# ============================================================================
results_df = pd.DataFrame({
    'Model': [key.rsplit('_', 1)[0] for key in results.keys()],
    'Target': [f"target_{key.rsplit('_', 1)[1]}" for key in results.keys()],
    'Test_Accuracy': [result['test_accuracy'] for result in results.values()],
    'Test_F1_Macro': [result['test_f1_macro'] for result in results.values()],
    'Test_F1_Micro': [result['test_f1_micro'] for result in results.values()],
    'Method': ['SMOTE + Extreme Weights'] * len(results)
})
results_df.to_csv('model_results_final.csv', index=False)
print("\nSaved: model_results_final.csv")

# Create confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Confusion Matrices - Final XGBoost + SVM', fontsize=16, fontweight='bold')

plot_configs = [
    ('XGBoost_human', axes[0, 0]),
    ('XGBoost_asv', axes[0, 1]),
    ('SVM_human', axes[1, 0]),
    ('SVM_asv', axes[1, 1])
]

for key, ax in plot_configs:
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
plt.savefig('confusion_matrices_final.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrices_final.png")

# Save models
with open('trained_models_final.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Saved: trained_models_final.pkl")

print("\n" + "="*80)
print("STEP 4 FINAL COMPLETE")
print("="*80)
print("\nFinal approach: XGBoost + SVM with SMOTE + Extreme Weights (100x lamb)")
print("Files created:")
print("  - model_results_final.csv")
print("  - confusion_matrices_final.png")
print("  - trained_models_final.pkl")