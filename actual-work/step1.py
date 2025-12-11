

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#the three datasets
print("datasets...")

try:
    df_orig = pd.read_csv('system_Original.csv')
    df_X = pd.read_csv('system_X.csv')
    df_Y = pd.read_csv('system_Y.csv')
    
    #system identifier so i can track which system each sample came from
    df_orig['system'] = 'orig'
    df_X['system'] = 'X'
    df_Y['system'] = 'Y'
    
    print(f" Loaded system_orig: {df_orig.shape[0]} samples, {df_orig.shape[1]} columns")

    print(f" Loaded system_X: {df_X.shape[0]} samples, {df_X.shape[1]} columns")
    print(f" Loaded system_Y: {df_Y.shape[0]} samples, {df_Y.shape[1]} columns")
    
except FileNotFoundError as e:
    print(f"\n ERROR: Could not find data files!")
    print(f"   Make sure system_Original.csv, system_X.csv, system_Y.csv are in this folder")
    print(f"   Error details: {e}")
    exit(1)

# column analysis

# Get coloumn names
feature_cols = [col for col in df_orig.columns 
                if col not in ['sid', 'uid', 'gender', 'target_human', 'target_asv', 'filename', 'system']]

print(f"\nTotal columns: {len(df_orig.columns)}")
print(f"Feature columns : {len(feature_cols)}")
print(f"Label/metadata columns: sid, uid, gender, target_human, target_asv, filename")
print(f"\nfirst 5 feature names: {feature_cols[:5]}")

# Speaker analysis

speakers = df_orig['sid'].unique()
print(f"\nTotal unique speakers: {len(speakers)}")
print(f"Speaker IDs: {sorted(speakers)}")

# Gender breakdown
gender_counts = df_orig.groupby('sid')['gender'].first().value_counts()
print(f"\nGender distribution:")
print(f"Male speakers: {gender_counts.get('M', 0)}")
print(f"Female speakers: {gender_counts.get('F', 0)}")

# Samples per speaker

samples_per_speaker = df_orig.groupby('sid').size()
print(f"\nSamples per speaker (system_orig only):")
print(f" Mean: {samples_per_speaker.mean():.1f}")
print(f" Min: {samples_per_speaker.min()}")
print(f" Max: {samples_per_speaker.max()}")
print(f"Std: {samples_per_speaker.std():.1f}")

#if all systems are together
total_samples_per_speaker = pd.concat([df_orig, df_X, df_Y]).groupby('sid').size()
print(f"\nIf we COMBINE all three systems:")
print(f"Total samples per speaker:")
print(f"Mean: {total_samples_per_speaker.mean():.1f}")
print(f" Min: {total_samples_per_speaker.min()}")
print(f"Max: {total_samples_per_speaker.max()}")
print(f"Total samples : {total_samples_per_speaker.sum()}")

# distribution analysis

print("\nClass distributions:")
human_dist = df_orig['target_human'].value_counts().sort_index()
asv_dist = df_orig['target_asv'].value_counts().sort_index()
print("target_human:", dict(human_dist))
print("target_asv:", dict(asv_dist))

#  imbalance ratio
human_max = human_dist.max()
human_min = human_dist.min()
asv_max = asv_dist.max()
asv_min = asv_dist.min()

print(f"\nImbalance ratios:")
print(f"target_human: {human_max/human_min:.2f}x")
print(f"target_asv: {asv_max/asv_min:.2f}x")

# missing value check

missing_orig = df_orig.isnull().sum().sum()
missing_X = df_X.isnull().sum().sum()
missing_Y = df_Y.isnull().sum().sum()

print(f"\nmissin values:")
print(f"system_orig: {missing_orig}")
print(f"ssystem_X: {missing_X}")
print(f"ssystem_Y: {missing_Y}")

if missing_orig + missing_X + missing_Y > 0:
    missing_cols = df_orig.columns[df_orig.isnull().any()].tolist()
    if missing_cols:
        print(f"columns with missing values: {missing_cols}")

# System combination analysis
total_samples = df_orig.shape[0] + df_X.shape[0] + df_Y.shape[0]
print(f"\nTotal samples when combined: {total_samples}")

# feature statistics
example_features = feature_cols[:5]
print(f"\nFeature statistics (first 5):")
print(df_orig[example_features].describe())
