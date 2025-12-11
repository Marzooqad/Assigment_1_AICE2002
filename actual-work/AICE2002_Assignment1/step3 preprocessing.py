import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#random seed agin
# == reproducibility

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Loading...")

train_df = pd.read_csv('train_data.csv')
valid_df = pd.read_csv('valid_data.csv')
test_df = pd.read_csv('test_data.csv')

print(f"Loaded training set: {train_df.shape[0]} samples, {train_df.shape[1]} columns")
print(f"Loaded validation set: {valid_df.shape[0]} samples, {valid_df.shape[1]} columns")
print(f"Loaded test set: {test_df.shape[0]} samples, {test_df.shape[1]} columns")

# seprate features from labels

#  which columns are features vs labels

metadata_cols = ['sid', 'uid', 'gender', 'filename', 'system']
label_cols = ['target_human', 'target_asv']
feature_cols = [col for col in train_df.columns 
                if col not in metadata_cols + label_cols]

print(f"Total feature columns: {len(feature_cols)}")
print(f"Label columns: {label_cols}")
print(f"Metadata columns: {metadata_cols}")

# pull features and labels for all splits
X_train = train_df[feature_cols].values
y_train_human = train_df['target_human'].values
y_train_asv = train_df['target_asv'].values

X_valid = valid_df[feature_cols].values
y_valid_human = valid_df['target_human'].values
y_valid_asv = valid_df['target_asv'].values

X_test = test_df[feature_cols].values
y_test_human = test_df['target_human'].values
y_test_asv = test_df['target_asv'].values

print(f"\nFeature matrix shape:")
print(f" X_train: {X_train.shape}")
print(f" X_valid: {X_valid.shape}")
print(f" X_test: {X_test.shape}")

#missing values??

train_missing = np.isnan(X_train).sum()
valid_missing = np.isnan(X_valid).sum()
test_missing = np.isnan(X_test).sum()

print(f"\nMissing values detected:")
print(f"  Training set: {train_missing} missing values")
print(f"  Validation set: {valid_missing} missing values")
print(f"  Test set: {test_missing} missing values")

if train_missing + valid_missing + test_missing > 0:
    print("missing values detected. strategy needed.")

# FINALLY StandardScaler
print("\nApplying StandardScaler...")

# fit scaler on training data only (no leaks)
scaler = StandardScaler()
scaler.fit(X_train)

# ransform all three sets using the training set statistics
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

print(f"\nScaling statistics computed from training set:")
print(f" Feature means: min={scaler.mean_.min():.2f}, max={scaler.mean_.max():.2f}")
print(f" Feature std devs: min={scaler.scale_.min():.2f}, max={scaler.scale_.max():.2f}")

# Verify scaling
print(f"Scaled features _ Mean: {X_train_scaled.mean(axis=0).mean():.6f}, Std: {X_train_scaled.std(axis=0).mean():.6f}")

# Analyze class imbalance

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


# saves preprocessed data

# Save scaled features
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_valid_scaled.npy', X_valid_scaled)
np.save('X_test_scaled.npy', X_test_scaled)

# but this is labels
np.save('y_train_human.npy', y_train_human)
np.save('y_train_asv.npy', y_train_asv)
np.save('y_valid_human.npy', y_valid_human)
np.save('y_valid_asv.npy', y_valid_asv)
np.save('y_test_human.npy', y_test_human)
np.save('y_test_asv.npy', y_test_asv)

# scaler for future use 
# potentially ( I did use it again)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# feature names for reference
with open('feature_names.txt', 'w') as f:
    for feature in feature_cols:
        f.write(feature + '\n')

print("Preprocessed data saved successfully")

# Create visualizations
