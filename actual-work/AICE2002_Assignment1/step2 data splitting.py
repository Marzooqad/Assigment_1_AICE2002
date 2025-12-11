

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# set random seed
#reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("combining datasets...")

# Load all three systems
df_orig = pd.read_csv('system_Original.csv')
df_X = pd.read_csv('system_X.csv')
df_Y = pd.read_csv('system_Y.csv')

# Add system identifier 
# track which system each sample came from where
df_orig['system'] = 'orig'
df_X['system'] = 'X'
df_Y['system'] = 'Y'

# one large dataset
df_combined = pd.concat([df_orig, df_X, df_Y], ignore_index=True)

print(f" Combined dataset: {df_combined.shape[0]} samples, {df_combined.shape[1]} columns")
print(f"  - system_orig: {len(df_orig)} samples")
print(f"  - system_X: {len(df_X)} samples")
print(f"  - system_Y: {len(df_Y)} samples")

#  speaker based split

#  unique speakers ig
unique_speakers = df_combined['sid'].unique()
print(f" Total unique speakers: {len(unique_speakers)}")

# datarame??
speaker_info = df_combined.groupby('sid').agg({
    'gender': 'first',  # gender for each speaker
    'sid': 'count'       # samples per speaker
}).rename(columns={'sid': 'n_samples'})

print(f"\nSpeaker information (samples per speaker): ")
print(f"mean: {speaker_info['n_samples'].mean():.1f}")
print(f"range: {speaker_info['n_samples'].min()} - {speaker_info['n_samples'].max()}")

# gender distribution
gender_counts = speaker_info['gender'].value_counts()
print(f"\nGender distribution among speakers:")
for gender, count in gender_counts.items():
    print(f"  {gender}: {count} speakers")

# THE BIG DIVIDE
# train/valid/test sets

# first split- separate test speakers (15% ≈ 4 speakers)
train_val_speakers, test_speakers = train_test_split(
    speaker_info.index,
    test_size=0.15,  # 15% for test
    random_state=RANDOM_STATE,
    stratify=speaker_info['gender'] if len(gender_counts) > 1 else None  # Stratify if multiple genders
)

# Second split: separate validation speakers from train (15% of total ≈ 5 speakers)
train_speakers, valid_speakers = train_test_split(
    train_val_speakers,
    test_size=0.176,  # this gives ~15% of original total
    random_state=RANDOM_STATE,
    stratify=speaker_info.loc[train_val_speakers, 'gender'] if len(gender_counts) > 1 else None
)

print(f" Train speakers: {len(train_speakers)} ({len(train_speakers)/len(unique_speakers)*100:.1f}%)")
print(f" Valid speakers: {len(valid_speakers)} ({len(valid_speakers)/len(unique_speakers)*100:.1f}%)")
print(f" Test speakers: {len(test_speakers)} ({len(test_speakers)/len(unique_speakers)*100:.1f}%)")

# create the train/valid/test datasets

# split the combined dataframe based on speaker assignments
train_df = df_combined[df_combined['sid'].isin(train_speakers)].copy()
valid_df = df_combined[df_combined['sid'].isin(valid_speakers)].copy()
test_df = df_combined[df_combined['sid'].isin(test_speakers)].copy()

print(f" Train set: {len(train_df)} samples ({len(train_df)/len(df_combined)*100:.1f}%)")
print(f" Valid set: {len(valid_df)} samples ({len(valid_df)/len(df_combined)*100:.1f}%)")
print(f" Test set: {len(test_df)} samples ({len(test_df)/len(df_combined)*100:.1f}%)")

# no speaker appears in multiple splits?
train_sids = set(train_df['sid'].unique())
valid_sids = set(valid_df['sid'].unique())
test_sids = set(test_df['sid'].unique())

overlap_train_valid = train_sids & valid_sids
overlap_train_test = train_sids & test_sids
overlap_valid_test = valid_sids & test_sids

if len(overlap_train_valid) > 0 or len(overlap_train_test) > 0 or len(overlap_valid_test) > 0:
    print("WARNING: Speaker overlap detected!")
    print(f"Train-Valid: {overlap_train_valid}")
    print(f"Train-Test: {overlap_train_test}")
    print(f"Valid-Test: {overlap_valid_test}")

# analyses class distribution across splits

def print_class_distribution(df, split_name, target_col):
    """Helper function to print class distribution"""
    dist = df[target_col].value_counts().sort_index()
    print(f"\n{split_name} - {target_col}:")
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {label:6s}: {count:4d} samples ({pct:5.1f}%)")
    return dist

train_human = print_class_distribution(train_df, "TRAIN", "target_human")
valid_human = print_class_distribution(valid_df, "VALID", "target_human")
test_human = print_class_distribution(test_df, "TEST", "target_human")

train_asv = print_class_distribution(train_df, "TRAIN", "target_asv")
valid_asv = print_class_distribution(valid_df, "VALID", "target_asv")
test_asv = print_class_distribution(test_df, "TEST", "target_asv")

# class imbalance?

def check_imbalance(dist, target_name, split_name):
    """Check imbalance ratio"""
    max_count = dist.max()
    min_count = dist.min()
    ratio = max_count / min_count
    print(f"\n{split_name} - {target_name}:")
    print(f"  Most common class: {max_count} samples")
    print(f"  Least common class: {min_count} samples")
    print(f"  Imbalance ratio: {ratio:.2f}")
    
    if ratio > 3.0:
        print(f"    Severe imbalance")
    elif ratio > 1.5:
        print(f"    Moderate imbalance")
    
    return ratio

# imbalance for training set (importan!)
train_human_ratio = check_imbalance(train_human, "target_human", "TRAIN")
train_asv_ratio = check_imbalance(train_asv, "target_asv", "TRAIN")

# save splits to CSV files

# sprate features from labels
feature_cols = [col for col in df_combined.columns 
                if col not in ['sid', 'uid', 'gender', 'target_human', 'target_asv', 'filename', 'system']]

# save training set
train_df.to_csv('train_data.csv', index=False)
valid_df.to_csv('valid_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

speaker_splits = pd.DataFrame({
    'speaker_id': list(train_speakers) + list(valid_speakers) + list(test_speakers),
    'split': ['train']*len(train_speakers) + ['valid']*len(valid_speakers) + ['test']*len(test_speakers)
})
speaker_splits.to_csv('speaker_splits.csv', index=False)
print("Data splits saved successfully")
