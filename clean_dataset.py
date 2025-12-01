import pandas as pd
import os

# Files
INPUT_FILE = "Crop and fertilizer dataset.csv"
OUTPUT_FILE = "Crop_and_fertilizer_cleaned.csv"

def clean_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Could not find '{INPUT_FILE}'")
        return

    print(f"🧹 Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 1. Remove the useless 'Link' column
    if 'Link' in df.columns:
        df = df.drop('Link', axis=1)
        print("   - Dropped 'Link' column")

    # 2. Fix the whitespace bug in Soil_color ("Red " -> "Red")
    if 'Soil_color' in df.columns:
        df['Soil_color'] = df['Soil_color'].str.strip()
        print("   - Fixed whitespace in 'Soil_color'")

    # 3. Remove any accidental duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"   - Removed {removed_rows} duplicate rows")

    # Save the new file
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Success! Cleaned data saved to '{OUTPUT_FILE}'")
    print(f"   - Old Rows: {initial_rows}")
    print(f"   - New Rows: {len(df)}")

if __name__ == "__main__":
    clean_dataset()