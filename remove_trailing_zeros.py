import pandas as pd

# --- 1. Read the CSV ---
input_file = "Production/Geocoded/mmhctotalcounts_geocodedupload.csv"
output_file = "Production/Geocoded/mmhctotalcounts_geocodedupload.csv"

df = pd.read_csv(input_file)

# --- 2. Columns that need to be cleaned ---
id_cols = ["totalServed"]

for col in id_cols:
    if col in df.columns:
        # Convert values like 12345.0 → 12345 (int)
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    else:
        print(f"⚠️ Column '{col}' not found in the file.")

# --- 3. Save cleaned CSV ---
df.to_csv(output_file, index=False)
print(f"✅ Cleaned file saved as: {output_file}")