import pandas as pd
import numpy as np
import sys

# ML Libraries
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load Data
DATA_FILE = "Crop_and_fertilizer_cleaned.csv"
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows from {DATA_FILE}")
except FileNotFoundError:
    print(f"❌ Error: {DATA_FILE} not found. Run clean_data.py first.")
    sys.exit(1)

X = df.drop(['Fertilizer', 'Link'], axis=1, errors='ignore')
y = df['Fertilizer']

# 2. Define Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['District_Name', 'Soil_color', 'Crop'])
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# 3. Define Search Space
# We include 'None' for max_depth to allow the trees to grow fully (often best for Random Forest)
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

# 4. Define Cross-Validation Strategy (THE FIX!)
# shuffle=True ensures we mix Pune, Kolhapur, etc. together before splitting
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("\n🧪 Starting Grid Search (with Shuffling enabled)...")
grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, verbose=1, n_jobs=-1)
grid_search.fit(X, y)

print("\n✅ Optimization Complete!")
print(f"Best Accuracy: {grid_search.best_score_:.2%}")
print("Best Parameters:")
print(grid_search.best_params_)