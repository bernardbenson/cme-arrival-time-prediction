import pandas as pd

# Fix CME 01 which has spaces in column names
file_path = 'data/01_2010-04-03/Train_01_2010-04-03_AB_NN.txt'

print(f"Fixing {file_path}...")

# Read the file
df = pd.read_csv(file_path)

print(f"Original columns: {list(df.columns)}")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

print(f"Cleaned columns: {list(df.columns)}")

# Save back to file
df.to_csv(file_path, index=False)

print(f"✓ Fixed column names in {file_path}")
