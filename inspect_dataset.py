import pandas as pd

file_path = 'D:\mini\Crop_and_fertilizer_cleaned.csv'

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Failed loading CSV: {e}")
    raise

out_lines = []
out_lines.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
out_lines.append('\nColumns:')
out_lines.append(', '.join(list(df.columns)))

out_lines.append('\nDtypes:')
out_lines.append(df.dtypes.to_string())

out_lines.append('\nMissing values (per column):')
out_lines.append(df.isnull().sum().to_string())

out_lines.append('\nSample rows:')
out_lines.append(df.head(5).to_string(index=False))

# Numeric summary
num = df.select_dtypes(include=['number'])
if not num.empty:
    out_lines.append('\nNumeric summary (describe):')
    out_lines.append(num.describe().T.to_string())
else:
    out_lines.append('\nNo numeric columns found.')

# Categorical summary: unique counts + top values
cat = df.select_dtypes(include=['object', 'category'])
if not cat.empty:
    out_lines.append('\nCategorical columns summary:')
    for c in cat.columns:
        try:
            vc = df[c].value_counts(dropna=False)
            out_lines.append(f"\nColumn '{c}': unique={df[c].nunique(dropna=False)}")
            out_lines.append(vc.head(10).to_string())
        except Exception as e:
            out_lines.append(f"Could not summarize column {c}: {e}")

summary_text = '\n'.join(out_lines)
print(summary_text)

# Save to file for quick reference
with open('dataset_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)

print('\nSaved summary to dataset_summary.txt')
