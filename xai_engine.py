import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

def explain_prediction(pipeline, user_input_df):
    print("   [System] 🧠 Running XAI Analysis (Why this result?)...")
    warnings.filterwarnings('ignore')

    try:
        model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # 1. Transform Input
        processed_input = preprocessor.transform(user_input_df)
        
        # Handle sparse matrix
        if hasattr(processed_input, "toarray"):
            processed_input = processed_input.toarray()
            
        # 2. Get Real Feature Names
        # We try to get names from the preprocessor (requires scikit-learn > 1.0)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback logic if method varies by version
            feature_names = [] 
            # (In a real scenario we'd rebuild them manually here, but most envs are new enough)
            print("   [Warning] Could not fetch feature names automatically.")

        # 3. Clean up the names
        # The system adds prefixes like 'num__Nitrogen' or 'cat__District_Kolhapur'
        # We remove them to make it look nice.
        clean_names = [
            name.replace('num__', '').replace('cat__', '').replace('remainder__', '') 
            for name in feature_names
        ]

        # 4. Calculate SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed_input)
        
        prediction_idx = model.predict(processed_input)[0] 
        class_idx = list(model.classes_).index(prediction_idx)
        
        # Extract specific class data
        shap_data = None
        if isinstance(shap_values, list):
            shap_data = shap_values[class_idx]
        else:
            if len(shap_values.shape) == 3:
                shap_data = shap_values[:, :, class_idx]
            else:
                shap_data = shap_values

        # 5. Plot with REAL NAMES
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_data, 
            processed_input, 
            feature_names=clean_names,  # <--- Passing the real names here
            show=False, 
            plot_type="bar"
        )
        
        plt.title(f"Main Factors for Recommending: {prediction_idx}", fontsize=12)
        plt.tight_layout()
        plt.savefig("explanation_shap.png")
        plt.close() # Free memory
        print(f"   [System] 💡 Explanation saved to 'explanation_shap.png'")
        
    except Exception as e:
        print(f"   [Warning] XAI Skipped: {e}")
        import traceback
        traceback.print_exc()