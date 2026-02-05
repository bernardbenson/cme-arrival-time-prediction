import pandas as pd

# List of CMEs that have the EA_raw column
cmes_to_fix = [
    'data/06_2011-10-22/Train_06_2011-10-22_AB_NN.txt',
    'data/10_2012-07-03/Train_10_2012-07-03_AB_NN.txt',
    'data/13_2012-10-05/Train_13_2012-10-05_AB_NN.txt',
    'data/15_2013-06-30/Train_15_2013-06-30_AB_NN.txt'
]

for file_path in cmes_to_fix:
    print(f"Processing {file_path}...")

    # Read the file
    df = pd.read_csv(file_path)

    # Check if EA_raw column exists
    if 'EA_raw' in df.columns:
        # Remove the EA_raw column
        df = df.drop(columns=['EA_raw'])

        # Save back to file
        df.to_csv(file_path, index=False)
        print(f"  ✓ Removed EA_raw column from {file_path}")
    else:
        print(f"  - EA_raw column not found in {file_path}")

    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    print()

print("Done!")
