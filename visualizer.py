import matplotlib.pyplot as plt
import numpy as np

def visualize_impact(user_input, recommended_fertilizer):
    """
    Generates a comparison graph showing Soil Nutrients BEFORE vs AFTER application.
    """
    print("\n   [System] 🎨 Generating Impact Report...")
    
    # 1. Get Current Data
    nutrients = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    current_values = [
        user_input.get('Nitrogen', 0), 
        user_input.get('Phosphorus', 0), 
        user_input.get('Potassium', 0)
    ]
    
    # 2. Simulate "After" Data (Heuristic Logic)
    # We estimate the boost based on the fertilizer type
    after_values = list(current_values)
    
    fert_name = recommended_fertilizer.lower()
    
    if 'urea' in fert_name:
        after_values[0] += 50  # Urea adds mostly Nitrogen
    elif 'dap' in fert_name:
        after_values[0] += 18
        after_values[1] += 46  # DAP is high in P
    elif 'mop' in fert_name:
        after_values[2] += 60  # MOP is Potassium
    elif 'ssp' in fert_name:
        after_values[1] += 16  # SSP is Phosphorus
    elif 'npk' in fert_name:
        # Generic boost for balanced fertilizers
        after_values[0] += 20
        after_values[1] += 20
        after_values[2] += 20
    else:
        # General maintainer
        after_values = [x + 5 for x in after_values]

    # 3. Create Plot
    plt.figure(figsize=(12, 6))
    
    # Chart settings
    x = np.arange(len(nutrients))
    width = 0.35
    
    # Plot Bars
    plt.bar(x - width/2, current_values, width, label='Current Soil Status', color='#ff9999', edgecolor='black')
    plt.bar(x + width/2, after_values, width, label='Expected After 1 Month', color='#99ff99', edgecolor='black')
    
    # Labels and Titles
    plt.ylabel('Nutrient Level (kg/ha)')
    plt.title(f'Predicted Impact of {recommended_fertilizer} Application', fontsize=14, fontweight='bold')
    plt.xticks(x, nutrients)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value tags on bars
    for i in range(len(current_values)):
        plt.text(i - width/2, current_values[i] + 1, str(current_values[i]), ha='center', fontsize=10)
        plt.text(i + width/2, after_values[i] + 1, str(after_values[i]), ha='center', fontsize=10, fontweight='bold')

    # Save and Show
    output_file = "impact_analysis.png"
    plt.savefig(output_file)
    print(f"   [System] 📸 Graph saved as '{output_file}'")
    
    # Note: If running on a server, we just save. If local, you can use plt.show()
    # plt.show()