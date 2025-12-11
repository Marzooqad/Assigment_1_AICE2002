
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

#load data
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

# classifiers

# intilise classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
}

print("\nClassifiers initialized with baseline parameters.")

# SMOTE again :(

results = {}

def train_and_evaluate(clf, clf_name, X_train, y_train, X_valid, y_valid, X_test, y_test, target_name):
    """train classifier with SMOTE and evaluation"""
    print(f"Training {clf_name} on {target_name}...")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"  Training samples: {len(y_train)} -> {len(y_train_resampled)}")
    
    # train on resampled data
    start_time = time.time()
    clf.fit(X_train_resampled, y_train_resampled)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time:.2f}s")
    
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
    
    # Per class metrics on test set
    test_precision = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    
    # Confusion matrix on test set
    cm = confusion_matrix(y_test, y_test_pred)
    
    #  class labels
    classes = np.unique(np.concatenate([y_train, y_valid, y_test]))
    
    print(f" testing Accuracy: {test_accuracy:.4f}, F1-macro: {test_f1_macro:.4f}")
    
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

#  all models on both targets
for clf_name, clf in classifiers.items():
    print(f"\n{clf_name}:")
    
    # Train on target_human
    results[f'{clf_name}_human'] = train_and_evaluate(
        clf, clf_name, X_train, y_train_human, 
        X_valid, y_valid_human, X_test, y_test_human, 
        'target_human'
    )
    
    # Re initialize classifier for target_asv (vv important!)
    if clf_name == 'Random Forest':
        clf_asv = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )
    else:  # SVM
        clf_asv = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
    
    # train on target_asv
    results[f'{clf_name}_asv'] = train_and_evaluate(
        clf_asv, clf_name, X_train, y_train_asv,
        X_valid, y_valid_asv, X_test, y_test_asv,
        'target_asv'
    )

# Results summary
for key, result in results.items():
    model_name, target = key.rsplit('_', 1)
    print(f"{model_name} - target_{target}: Acc={result['test_accuracy']:.4f}, F1={result['test_f1_macro']:.4f}")

# svae improved results
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

# Per class metrics

for key, result in results.items():
    model_name, target = key.rsplit('_', 1)
    print(f"\n{model_name} - target_{target}:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*50)
    
    for i, cls in enumerate(result['classes']):
        if i < len(result['test_precision']):
            recall_val = result['test_recall'][i]
            print(f"{cls:<10} P:{result['test_precision'][i]:.4f} R:{result['test_recall'][i]:.4f} F1:{result['test_f1'][i]:.4f}")
        else:
            print(f"{cls:<10} N/A (not in test set)")
