"""
AICE2002 Assignment 1 - STEP 2: Data Splitting (TASK 1)
Author: [Your Name]
Date: November 2, 2025

PURPOSE: Implement speaker-based stratified train/valid/test split
This addresses Task 1 of your assignment.

KEY DECISIONS (based on Step 1 exploration):
- Combine all 3 systems (4,923 total samples)
- Split by SPEAKER (not by sample) to prevent data leakage
- 21 speakers train / 5 valid / 4 test (70/15/15 split)
- Set random_state=42 for reproducibility
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n" + "="*80)
print("STEP 2: DATA SPLITTING (TASK 1)")
print("="*80 + "\n")

# ============================================================================
# LINES 1-30: Load and combine all three systems
# ============================================================================
print("Loading and combining datasets...")

# Load all three systems
df_orig = pd.read_csv('system_Original.csv')
df_X = pd.read_csv('system_X.csv')
df_Y = pd.read_csv('system_Y.csv')

# Add system identifier to track which system each sample came from
df_orig['system'] = 'orig'
df_X['system'] = 'X'
df_Y['system'] = 'Y'

# Combine into one large dataset
df_combined = pd.concat([df_orig, df_X, df_Y], ignore_index=True)

print(f" Combined dataset: {df_combined.shape[0]} samples, {df_combined.shape[1]} columns")
print(f"  - system_orig: {len(df_orig)} samples")
print(f"  - system_X: {len(df_X)} samples")
print(f"  - system_Y: {len(df_Y)} samples")

# ============================================================================
# LINES 31-60: Prepare speaker information for stratified splitting
# ============================================================================
print("\nPreparing speaker-based split...")

# Get unique speakers
unique_speakers = df_combined['sid'].unique()
print(f" Total unique speakers: {len(unique_speakers)}")

# Create speaker-level dataframe for splitting
speaker_info = df_combined.groupby('sid').agg({
    'gender': 'first',  # Get gender for each speaker
    'sid': 'count'       # Count samples per speaker
}).rename(columns={'sid': 'n_samples'})

print(f"\nSpeaker information:")
print(f"  Samples per speaker (mean): {speaker_info['n_samples'].mean():.1f}")
print(f"  Samples per speaker (range): {speaker_info['n_samples'].min()} - {speaker_info['n_samples'].max()}")

# Check gender distribution
gender_counts = speaker_info['gender'].value_counts()
print(f"\nGender distribution among speakers:")
for gender, count in gender_counts.items():
    print(f"  {gender}: {count} speakers")

# ============================================================================
# LINES 61-100: Split speakers into train/valid/test sets
# ============================================================================
print(f"\nSplitting speakers...")

# First split: separate test speakers (15% ≈ 4 speakers)
train_val_speakers, test_speakers = train_test_split(
    speaker_info.index,
    test_size=0.15,  # 15% for test
    random_state=RANDOM_STATE,
    stratify=speaker_info['gender'] if len(gender_counts) > 1 else None  # Stratify if multiple genders
)

# Second split: separate validation speakers from train (15% of total ≈ 5 speakers)
# 15% of total = 15/(100-15) = 17.6% of train_val
train_speakers, valid_speakers = train_test_split(
    train_val_speakers,
    test_size=0.176,  # This gives us ~15% of original total
    random_state=RANDOM_STATE,
    stratify=speaker_info.loc[train_val_speakers, 'gender'] if len(gender_counts) > 1 else None
)

print(f" Train speakers: {len(train_speakers)} ({len(train_speakers)/len(unique_speakers)*100:.1f}%)")
print(f" Valid speakers: {len(valid_speakers)} ({len(valid_speakers)/len(unique_speakers)*100:.1f}%)")
print(f" Test speakers: {len(test_speakers)} ({len(test_speakers)/len(unique_speakers)*100:.1f}%)")

# ============================================================================
# LINES 101-130: Create train/valid/test datasets
# ============================================================================
print(f"\nCreating train/valid/test splits based on speakers...")

# Split the combined dataframe based on speaker assignments
train_df = df_combined[df_combined['sid'].isin(train_speakers)].copy()
valid_df = df_combined[df_combined['sid'].isin(valid_speakers)].copy()
test_df = df_combined[df_combined['sid'].isin(test_speakers)].copy()

print(f" Train set: {len(train_df)} samples ({len(train_df)/len(df_combined)*100:.1f}%)")
print(f" Valid set: {len(valid_df)} samples ({len(valid_df)/len(df_combined)*100:.1f}%)")
print(f" Test set: {len(test_df)} samples ({len(test_df)/len(df_combined)*100:.1f}%)")

# Verify no speaker appears in multiple splits
train_sids = set(train_df['sid'].unique())
valid_sids = set(valid_df['sid'].unique())
test_sids = set(test_df['sid'].unique())

overlap_train_valid = train_sids & valid_sids
overlap_train_test = train_sids & test_sids
overlap_valid_test = valid_sids & test_sids

if len(overlap_train_valid) == 0 and len(overlap_train_test) == 0 and len(overlap_valid_test) == 0:
    print(" VERIFIED: No speaker overlap between splits (no data leakage!)")
else:
    print("    WARNING: Speaker overlap detected!")
    print(f"   Train-Valid overlap: {overlap_train_valid}")
    print(f"   Train-Test overlap: {overlap_train_test}")
    print(f"   Valid-Test overlap: {overlap_valid_test}")

# ============================================================================
# LINES 131-180: Analyze class distribution across splits
# ============================================================================
print("\n" + "="*80)
print("CLASS DISTRIBUTION ANALYSIS (Critical for Task 2!)")
print("="*80)

def print_class_distribution(df, split_name, target_col):
    """Helper function to print class distribution"""
    dist = df[target_col].value_counts().sort_index()
    print(f"\n{split_name} - {target_col}:")
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {label:6s}: {count:4d} samples ({pct:5.1f}%)")
    return dist

# Analyze target_human distribution
print("\n--- TARGET_HUMAN Distribution Across Splits ---")
train_human = print_class_distribution(train_df, "TRAIN", "target_human")
valid_human = print_class_distribution(valid_df, "VALID", "target_human")
test_human = print_class_distribution(test_df, "TEST", "target_human")

# Analyze target_asv distribution
print("\n--- TARGET_ASV Distribution Across Splits ---")
train_asv = print_class_distribution(train_df, "TRAIN", "target_asv")
valid_asv = print_class_distribution(valid_df, "VALID", "target_asv")
test_asv = print_class_distribution(test_df, "TEST", "target_asv")

# ============================================================================
# LINES 181-220: Check for severe class imbalance (important for Task 2!)
# ============================================================================
print("\n" + "="*80)
print("IMBALANCE CHECK (Will you need to address this in Task 2?)")
print("="*80)

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
        print(f"    SEVERE IMBALANCE - Must address in Task 2!")
    elif ratio > 1.5:
        print(f"    MODERATE IMBALANCE - Should address in Task 2")
    else:
        print(f"   Balanced")
    
    return ratio

# Check imbalance for training set (most important!)
train_human_ratio = check_imbalance(train_human, "target_human", "TRAIN")
train_asv_ratio = check_imbalance(train_asv, "target_asv", "TRAIN")

# ============================================================================
# LINES 221-260: Save splits to CSV files
# ============================================================================
print("\n" + "="*80)
print("SAVING SPLITS TO FILES")
print("="*80)

# Separate features from labels
feature_cols = [col for col in df_combined.columns 
                if col not in ['sid', 'uid', 'gender', 'target_human', 'target_asv', 'filename', 'system']]

# Save train set
train_df.to_csv('train_data.csv', index=False)
print(f" Saved: train_data.csv ({len(train_df)} samples)")

# Save validation set
valid_df.to_csv('valid_data.csv', index=False)
print(f" Saved: valid_data.csv ({len(valid_df)} samples)")

# Save test set
test_df.to_csv('test_data.csv', index=False)
print(f" Saved: test_data.csv ({len(test_df)} samples)")

# Save speaker assignments for reference
speaker_splits = pd.DataFrame({
    'speaker_id': list(train_speakers) + list(valid_speakers) + list(test_speakers),
    'split': ['train']*len(train_speakers) + ['valid']*len(valid_speakers) + ['test']*len(test_speakers)
})
speaker_splits.to_csv('speaker_splits.csv', index=False)
print(f" Saved: speaker_splits.csv (speaker assignment tracking)")
