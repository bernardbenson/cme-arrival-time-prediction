import pandas as pd
import os
import glob

# Expected columns in correct order
expected_columns = ['Ensemble_member', 'Time_since_eruption', 'EA_diff_A', 'Travel_time_x', 'EA_diff_B', 'Travel_time_y']

# Find all training data files
data_files = sorted(glob.glob('data/*/Train_*_AB_NN.txt'))

print(f"Checking {len(data_files)} files...\n")
print("="*80)

all_valid = True

for file_path in data_files:
    file_name = os.path.basename(file_path)
    print(f"\n{file_name}")
    print("-" * 60)

    try:
        # Read the file
        df = pd.read_csv(file_path)

        # Check columns
        if list(df.columns) != expected_columns:
            print(f"  ❌ INCORRECT COLUMNS")
            print(f"     Expected: {expected_columns}")
            print(f"     Found:    {list(df.columns)}")
            all_valid = False
        else:
            print(f"  ✓ Columns correct: {len(df.columns)} columns")

        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"  ❌ FOUND NaN VALUES: {nan_count} total NaNs")
            print(f"     NaNs per column:")
            for col in df.columns:
                col_nans = df[col].isna().sum()
                if col_nans > 0:
                    print(f"       - {col}: {col_nans}")
            all_valid = False
        else:
            print(f"  ✓ No NaN values")

        # Check data types
        print(f"  ✓ Shape: {df.shape} (rows, columns)")

        # Check for any empty values or whitespace
        for col in df.columns:
            if df[col].dtype == 'object':
                empty_or_whitespace = df[col].apply(lambda x: str(x).strip() == '' if pd.notna(x) else False).sum()
                if empty_or_whitespace > 0:
                    print(f"  ⚠ Warning: {empty_or_whitespace} empty/whitespace values in {col}")
                    all_valid = False

    except Exception as e:
        print(f"  ❌ ERROR reading file: {e}")
        all_valid = False

print("\n" + "="*80)
if all_valid:
    print("✅ ALL FILES ARE VALID - Correct format with no NaN values!")
else:
    print("❌ SOME FILES HAVE ISSUES - See details above")
print("="*80)
