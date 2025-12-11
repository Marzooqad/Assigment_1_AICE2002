
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import time

#this one takes a bit of time to run 

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

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

print(f"train set: {X_train.shape}")
print(f"Validation set: {X_valid.shape}")
print(f"test set: {X_test.shape}")

# combine both train and validation for GridSearchCV
X_train_full = np.vstack([X_train, X_valid])
y_train_full_human = np.concatenate([y_train_human, y_valid_human])
y_train_full_asv = np.concatenate([y_train_asv, y_valid_asv])

print(f"Combined train+valid: {X_train_full.shape}")

# hyperparameter search spaces

# define parameter grids
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

# Hyperparameter tuning with GridSearchCV
print("Running GridSearchCV...")

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
    
    print(f"  Tuning completed in {tune_time:.1f}s")
    print(f"  Best CV F1-macro: {grid_search.best_score_:.4f}")
    print(f"  Best parameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"  Test Accuracy: {test_accuracy:.4f}, F1-macro: {test_f1_macro:.4f}")
    
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

# Compare baseline vs tuned performance
print("\nTuned Performance:")
for key, result in results_tuning.items():
    print(f"{result['model_name']} - {result['target']}: Acc={result['test_accuracy']:.4f}, F1={result['test_f1_macro']:.4f}")

try:
    baseline_results = pd.read_csv('model_results.csv')
    print("\nBaseline Performance:")
    for _, row in baseline_results.iterrows():
        print(f"{row['Model']} - {row['Target']}: Acc={row['Test_Accuracy']:.4f}, F1={row['Test_F1_Macro']:.4f}")
except:
    pass

# Calculate improvements
print("\nPerformance Improvements:")

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
        
        print(f"{model_name} - {target}: {improvement:+.4f} ({pct_improvement:+.1f}%)")
        
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

# best hyperparameters

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

print("complete!!!!")
