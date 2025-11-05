"""
AICE2002 Assignment 1 - STEP 1: Data Exploration
Author: [Your Name]
Date: November 2, 2025

PURPOSE: Understand the data before making ANY decisions
This informs ALL your Tasks (1-6)

INSTRUCTIONS:
1. Download system_Original.csv, system_X.csv, system_Y.csv from Moodle
2. Place them in the same folder as this file
3. Run: python step1_explore_data.py
4. Read ALL the output carefully
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*80)
print("STEP 1: DATA EXPLORATION - Understanding What We Have")
print("="*80 + "\n")


# Load the three datasets
print("Loading datasets...")

try:
    df_orig = pd.read_csv('system_Original.csv')
    df_X = pd.read_csv('system_X.csv')
    df_Y = pd.read_csv('system_Y.csv')
    
    # Add system identifier so we can track which system each sample came from
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

# ============================================================================
# QUESTION 1: What columns do we have?
# ============================================================================
print("\n" + "-"*80)
print("QUESTION 1: What are the columns in our data?")
print("-"*80)

# Get column names
feature_cols = [col for col in df_orig.columns 
                if col not in ['sid', 'uid', 'gender', 'target_human', 'target_asv', 'filename', 'system']]

print(f"\nTotal columns: {len(df_orig.columns)}")
print(f"Feature columns (for ML): {len(feature_cols)}")
print(f"Label/metadata columns: sid, uid, gender, target_human, target_asv, filename")
print(f"\nFirst 5 feature names: {feature_cols[:5]}")
print(f"Last 5 feature names: {feature_cols[-5:]}")

# ============================================================================
# QUESTION 2: How many speakers do we have? (Important for Task 1!)
# ============================================================================
print("\n" + "-"*80)
print("QUESTION 2: How many speakers? (Critical for data splitting!)")
print("-"*80)

speakers = df_orig['sid'].unique()
print(f"\nTotal unique speakers: {len(speakers)}")
print(f"Speaker IDs: {sorted(speakers)}")

# Gender breakdown
gender_counts = df_orig.groupby('sid')['gender'].first().value_counts()
print(f"\nGender distribution:")
print(f"  Male speakers: {gender_counts.get('M', 0)}")
print(f"  Female speakers: {gender_counts.get('F', 0)}")

# ============================================================================
# QUESTION 3: How many samples per speaker?
# ============================================================================
print("\n" + "-"*80)
print("QUESTION 3: How many samples does each speaker have?")
print("-"*80)

samples_per_speaker = df_orig.groupby('sid').size()
print(f"\nSamples per speaker (system_orig only):")
print(f"  Mean: {samples_per_speaker.mean():.1f}")
print(f"  Min: {samples_per_speaker.min()}")
print(f"  Max: {samples_per_speaker.max()}")
print(f"  Std: {samples_per_speaker.std():.1f}")

# If we combine all three systems
total_samples_per_speaker = pd.concat([df_orig, df_X, df_Y]).groupby('sid').size()
print(f"\nIf we COMBINE all three systems:")
print(f"  Total samples per speaker:")
print(f"    Mean: {total_samples_per_speaker.mean():.1f}")
print(f"    Min: {total_samples_per_speaker.min()}")
print(f"    Max: {total_samples_per_speaker.max()}")
print(f"    Total samples available: {total_samples_per_speaker.sum()}")

# ============================================================================
# QUESTION 4: CLASS DISTRIBUTION - Is it balanced? (Critical!)
# ============================================================================
print("\n" + "="*80)
print("QUESTION 4: CLASS DISTRIBUTION - Are our classes balanced?")
print("(This is CRITICAL - your professor mentioned this!)")
print("="*80)

print("\n--- TARGET_HUMAN Distribution ---")
human_dist = df_orig['target_human'].value_counts().sort_index()
print(human_dist)
print(f"\nProportions:")
for label, count in human_dist.items():
    print(f"  {label}: {count/len(df_orig)*100:.1f}%")

print("\n--- TARGET_ASV Distribution ---")
asv_dist = df_orig['target_asv'].value_counts().sort_index()
print(asv_dist)
print(f"\nProportions:")
for label, count in asv_dist.items():
    print(f"  {label}: {count/len(df_orig)*100:.1f}%")

# Check imbalance ratio
human_max = human_dist.max()
human_min = human_dist.min()
asv_max = asv_dist.max()
asv_min = asv_dist.min()

print(f"\nIMBALANCE ANALYSIS:")
print(f"  target_human - Ratio (most common / least common): {human_max/human_min:.2f}")
print(f"  target_asv - Ratio (most common / least common): {asv_max/asv_min:.2f}")
print(f"\n  Note: Ratio > 1.5 suggests class imbalance (you should address this in Task 2!)")

# ============================================================================
# QUESTION 5: Missing values?
# ============================================================================
print("\n" + "-"*80)
print("QUESTION 5: Do we have missing values?")
print("-"*80)

missing_orig = df_orig.isnull().sum().sum()
missing_X = df_X.isnull().sum().sum()
missing_Y = df_Y.isnull().sum().sum()

print(f"\nMissing values:")
print(f"  system_orig: {missing_orig}")
print(f"  system_X: {missing_X}")
print(f"  system_Y: {missing_Y}")

if missing_orig + missing_X + missing_Y == 0:
    print("   Great! No missing values to handle")
else:
    print("    You'll need to handle missing values in Task 2")
    # Show which columns have missing values
    missing_cols = df_orig.columns[df_orig.isnull().any()].tolist()
    if missing_cols:
        print(f"  Columns with missing values: {missing_cols}")

# ============================================================================
# QUESTION 6: Should we combine systems or keep them separate?
# ============================================================================
print("\n" + "="*80)
print("QUESTION 6: Should we COMBINE systems or keep them SEPARATE?")
print("(This is Task 4 - you need to justify your decision!)")
print("="*80)

print(f"\nOption A: Use only system_orig")
print(f"  Pros: Simpler, original unmodified audio")
print(f"  Cons: Only {df_orig.shape[0]} samples")

print(f"\nOption B: Combine all three systems")
print(f"  Pros: {df_orig.shape[0] + df_X.shape[0] + df_Y.shape[0]} samples (3x more data!)")
print(f"  Cons: Systems X and Y are AI-modified (potential domain shift)")

print(f"\nYour professor said: 'Make one big dataset and split them'")
print(f"  → This suggests combining is the expected approach")
print(f"  → More data = better model generalization")
print(f"  → You're testing if labels are predictable ACROSS different processing")

# ============================================================================
# QUESTION 7: Feature statistics (quick check)
# ============================================================================
print("\n" + "-"*80)
print("QUESTION 7: Quick feature statistics check")
print("-"*80)

# Pick a few example features to show
example_features = feature_cols[:5]
print(f"\nExample statistics for first 5 features:")
print(df_orig[example_features].describe())
