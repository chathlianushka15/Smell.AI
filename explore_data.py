import pandas as pd

df = pd.read_csv("data/molecules.csv")

print("Rows (molecules):", len(df))
print("Columns:", len(df.columns))
print()
print("First 5 column names:")
print(list(df.columns[:5]))
print()
print("First 3 rows, first 2 columns:")
print(df.iloc[:3, :2].to_string())