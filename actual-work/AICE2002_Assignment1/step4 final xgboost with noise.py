
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

# load preprocessed data
print("Loading...")

X_train = np.load('X_train_scaled.npy')
X_valid = np.load('X_valid_scaled.npy')
X_test = np.load('X_test_scaled.npy')

y_train_human = np.load('y_train_human.npy', allow_pickle=True)
y_train_asv = np.load('y_train_asv.npy', allow_pickle=True)
y_valid_human = np.load('y_valid_human.npy', allow_pickle=True)
y_valid_asv = np.load('y_valid_asv.npy', allow_pickle=True)
y_test_human = np.load('y_test_human.npy', allow_pickle=True)
y_test_asv = np.load('y_test_asv.npy', allow_pickle=True)

# noise will be added AFTER SMOTE in the train_and_evaluate function
# this way SMOTE works on clean data, then we add noise to help generalization
NOISE_STD = 0.05  # 5% of feature scale (since features are standardized, std=1)

print("\n" + "="*80)
print("NOISE EXPERIMENT: Will add Gaussian noise after SMOTE resampling")
print(f"Noise standard deviation: {NOISE_STD}")
print("="*80 + "\n")

# Define classifiers with EXTREME class weights
# same as before - noise will be added after SMOTE

# convert class labels to numeric for XGBoost, then map weights
def get_sample_weights(y, class_weight_dict):
    """class weight dictionary to sample weights"""
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
        C=10.0,  # increased from 1.0 for more complex boundaries
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

print("Classifiers initialized.")

# train function with SMOTE + sample weights + NOISE (added after SMOTE)

results = {}

def train_and_evaluate(clf, clf_name, X_train, y_train, X_valid, y_valid, 
                        X_test, y_test, target_name, class_weights):
    
    print(f"\n  Training {clf_name} on {target_name}...")
    
    #  original distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"    Original training distribution:")
    for label, count in zip(unique, counts):
        print(f"      {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    #  SMOTE again... on clean data first
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    #  resampled distribution
    unique_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
    print(f"    After SMOTE:")
    for label, count in zip(unique_resampled, counts_resampled):
        print(f"      {label}: {count} samples ({count/len(y_train_resampled)*100:.1f}%)")
    
    # ADD NOISE AFTER SMOTE - this way SMOTE works on clean data
    # then we add noise to the resampled data for regularization
    noise = np.random.normal(0, NOISE_STD, size=X_train_resampled.shape)
    X_train_resampled = X_train_resampled + noise
    print(f"    Added Gaussian noise (std={NOISE_STD}) to resampled training data")
    
    # for XGBoost:  labels to integers
    if clf_name == 'XGBoost':
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_resampled)
        y_valid_encoded = label_encoder.transform(y_valid)
        y_test_encoded = label_encoder.transform(y_test)
        
        #  sample weights for encoded labels
        # original class weights to encoded integers
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
    
    # train model :) with noise added after SMOTE
    start_time = time.time()
    
    if clf_name == 'XGBoost':
        clf.fit(X_train_resampled, y_train_encoded, sample_weight=sample_weights, verbose=False)
        # predict with encoded labels, then decode AGAIN
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
    print(f" Training completed in {train_time:.2f} seconds")
    
    # metrics (use original string labels)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1_micro = f1_score(y_test, y_test_pred, average='micro')
    
    test_precision = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_test, y_test_pred)
    classes = np.unique(np.concatenate([y_train, y_valid, y_test]))
    
    print(f"  Test, Accuracy: {test_accuracy:.4f}, F1-macro: {test_f1_macro:.4f}")
    
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

# Train all models - noise will be added inside train_and_evaluate AFTER SMOTE
print("\n" + "="*80)
print("TRAINING MODELS WITH NOISE (added after SMOTE)")
print("="*80)

for clf_name in ['XGBoost', 'SVM']:
    results[f'{clf_name}_human'] = train_and_evaluate(
        classifiers_human[clf_name], clf_name,
        X_train, y_train_human,  # clean data - noise added after SMOTE
        X_valid, y_valid_human,
        X_test, y_test_human,
        'target_human',
        class_weights_human
    )

for clf_name in ['XGBoost', 'SVM']:
    results[f'{clf_name}_asv'] = train_and_evaluate(
        classifiers_asv[clf_name], clf_name,
        X_train, y_train_asv,  # clean data - noise added after SMOTE
        X_valid, y_valid_asv,
        X_test, y_test_asv,
        'target_asv',
        class_weights_asv
    )

# Results summary

print(f"\n{'Model':<15} {'Target':<15} {'Accuracy':<12} {'F1-macro':<12}")
print("-"*60)
for key, result in results.items():
    model_name, target = key.rsplit('_', 1)
    print(f"{model_name:<15} {f'target_{target}':<15} "
            f"{result['test_accuracy']:.4f}       {result['test_f1_macro']:.4f}")

# Per-class metrics - LAMB PLEASEEEE (maybe noise helps?)
print("\n" + "-"*80)
print("PER-CLASS METRICS (Focus on LAMB detection - with noise):")
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
            
            # check if we detected ANY lamb
            if cls == 'lamb' and recall_val > 0.0:
                lamb_detected = True
                marker = " *** LAMB DETECTED! ***"
            elif cls in ['lamb', 'wolf'] and recall_val > 0.0:
                marker = " (IMPROVED!)"
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
    print("LIMITATION: Lamb class remains undetectable even with noise")

# save results
results_df = pd.DataFrame({
    'Model': [key.rsplit('_', 1)[0] for key in results.keys()],
    'Target': [f"target_{key.rsplit('_', 1)[1]}" for key in results.keys()],
    'Test_Accuracy': [result['test_accuracy'] for result in results.values()],
    'Test_F1_Macro': [result['test_f1_macro'] for result in results.values()],
    'Test_F1_Micro': [result['test_f1_micro'] for result in results.values()],
    'Method': ['SMOTE + Extreme Weights + Noise'] * len(results)
})
results_df.to_csv('model_results_noise.csv', index=False)
print("\nSaved: model_results_noise.csv")

print("\nDone! Noise experiment complete.")

