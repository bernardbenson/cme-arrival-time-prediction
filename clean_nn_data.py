import pandas as pd
import numpy as np
import os

# Expected columns for all NN files
expected_columns = ['Ensemble_member', 'Time_since_eruption', 'EA_diff_A',
                   'Travel_time_x', 'EA_diff_B', 'Travel_time_y']

# Files that need EA_raw removed
files_to_clean = {
    '06': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/06_2011-10-22/Train_06_2011-10-22_AB_NN.txt',
    '10': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/10_2012-07-03/Train_10_2012-07-03_AB_NN.txt',
    '13': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/13_2012-10-05/Train_13_2012-10-05_AB_NN.txt',
    '15': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/15_2013-06-30/Train_15_2013-06-30_AB_NN.txt'
}

# All NN files for verification
all_files = {
    '01': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/01_2010-04-03/Train_01_2010-04-03_AB_NN.txt',
    '02': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/02_2010-05-23/Train_02_2010-05-23_AB_NN.txt',
    '03': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/03_2010-08-01/Train_03_2010-08-01_AB_NN.txt',
    '04': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/04_2011-09-06/Train_04_2011-09-06_AB_NN.txt',
    '05': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/05_2011-09-13/Train_05_2011-09-13_AB_NN.txt',
    '06': files_to_clean['06'],
    '07': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/07_2012-01-19/Train_07_2012-01-19_AB_NN.txt',
    '09': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/09_2012-06-14/Train_09_2012-06-14_AB_NN.txt',
    '10': files_to_clean['10'],
    '11': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/11_2012-07-12/Train_11_2012-07-12_AB_NN.txt',
    '12': '/Users/bbenson/my_bbenson/cme_arrival_time_prediction/data/12_2012-09-27/Train_12_2012-09-27_AB_NN.txt',
    '13': files_to_clean['13'],
    '15': files_to_clean['15']
}

print("=" * 80)
print("CLEANING NN DATASET FILES")
print("=" * 80)

# Clean files with EA_raw column
for cme_id, filepath in files_to_clean.items():
    print(f"\nProcessing CME {cme_id}...")

    # Read the file to check its current structure
    df = pd.read_csv(filepath)
    print(f"  Current shape: {df.shape}")
    print(f"  Current columns: {list(df.columns)}")

    # Check if EA_raw column exists
    if 'EA_raw' in df.columns:
        print(f"  Removing 'EA_raw' column...")
        df = df.drop(columns=['EA_raw'])
    else:
        # If column name is not EA_raw, check for unnamed columns or extra columns
        extra_cols = [col for col in df.columns if col not in expected_columns]
        if extra_cols:
            print(f"  Found extra columns: {extra_cols}")
            print(f"  Removing extra columns...")
            df = df[expected_columns]
        else:
            print(f"  WARNING: No EA_raw or extra columns found, but file was flagged for cleaning")

    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"  Found {nan_count} NaN values")
        print(f"  NaN distribution:\n{df.isna().sum()}")
    else:
        print(f"  No NaN values found")

    # Save the cleaned file
    print(f"  Saving cleaned file...")
    df.to_csv(filepath, index=False)
    print(f"  New shape: {df.shape}")
    print(f"  New columns: {list(df.columns)}")
    print(f"  ✓ CME {cme_id} cleaned successfully")

print("\n" + "=" * 80)
print("VERIFYING ALL NN DATASET FILES")
print("=" * 80)

issues_found = []

for cme_id, filepath in all_files.items():
    print(f"\nVerifying CME {cme_id}...")

    df = pd.read_csv(filepath)

    # Check columns
    if list(df.columns) != expected_columns:
        issues_found.append(f"CME {cme_id}: Column mismatch. Expected {expected_columns}, got {list(df.columns)}")
        print(f"  ✗ Column structure mismatch")
        print(f"    Expected: {expected_columns}")
        print(f"    Got: {list(df.columns)}")
    else:
        print(f"  ✓ Columns correct")

    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        issues_found.append(f"CME {cme_id}: {nan_count} NaN values found")
        print(f"  ✗ {nan_count} NaN values found:")
        print(f"    {df.isna().sum()}")
    else:
        print(f"  ✓ No NaN values")

    # Check shape
    print(f"  Shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} columns)")

print("\n" + "=" * 80)
if issues_found:
    print("ISSUES FOUND:")
    for issue in issues_found:
        print(f"  - {issue}")
else:
    print("✓ ALL FILES VERIFIED SUCCESSFULLY - No issues found!")
print("=" * 80)
