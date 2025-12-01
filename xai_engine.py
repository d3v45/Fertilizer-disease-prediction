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
        processed_input = preprocessor.transform(user_input_df)
        
        if hasattr(processed_input, "toarray"):
            processed_input = processed_input.toarray()
            
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed_input)
        
        prediction_idx = model.predict(processed_input)[0] 
        class_idx = list(model.classes_).index(prediction_idx)
        
        # FIX: Check if list or array to avoid crash
        shap_data = None
        if isinstance(shap_values, list):
            shap_data = shap_values[class_idx] if class_idx < len(shap_values) else shap_values[0]
        else:
            # Handle 3D array (samples, features, classes)
            if len(shap_values.shape) == 3:
                shap_data = shap_values[:, :, class_idx]
            else:
                shap_data = shap_values

        plt.figure(figsize=(10, 5))
        shap.summary_plot(shap_data, processed_input, show=False, plot_type="bar")
        plt.title(f"Why {prediction_idx}?")
        plt.tight_layout()
        plt.savefig("explanation_shap.png")
        print(f"   [System] 💡 Explanation saved to 'explanation_shap.png'")
        
    except Exception as e:
        print(f"   [Warning] XAI Skipped: {e}")