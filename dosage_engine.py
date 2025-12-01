import math

class DosageCalculator:
    def __init__(self):
        # 1. Define Ideal NPK requirements (kg/hectare) for common crops
        # (These are heuristic averages for valid novelty demo)
        self.CROP_STANDARDS = {
            'rice':      {'N': 100, 'P': 40, 'K': 40},
            'wheat':     {'N': 120, 'P': 60, 'K': 40},
            'sugarcane': {'N': 250, 'P': 100, 'K': 100},
            'cotton':    {'N': 80,  'P': 40, 'K': 40},
            'maize':     {'N': 120, 'P': 60, 'K': 50},
            'general':   {'N': 100, 'P': 50, 'K': 50}
        }

        # 2. Define Fertilizer Compositions (Percentage of N-P-K)
        self.FERT_SPECS = {
            'urea':         {'N': 0.46, 'P': 0.00, 'K': 0.00},  # 46% Nitrogen
            'dap':          {'N': 0.18, 'P': 0.46, 'K': 0.00},  # 18% N, 46% P
            'mop':          {'N': 0.00, 'P': 0.00, 'K': 0.60},  # 60% Potassium
            'ssp':          {'N': 0.00, 'P': 0.16, 'K': 0.00},  # 16% Phosphorus
            '19:19:19 npk': {'N': 0.19, 'P': 0.19, 'K': 0.19},
        }

    def calculate_dosage(self, crop, current_nutrients, recommended_fert):
        """
        Returns: (Amount in kg/acre, Logic Message)
        """
        crop_key = crop.lower() if crop.lower() in self.CROP_STANDARDS else 'general'
        target = self.CROP_STANDARDS[crop_key]
        
        # Clean fertilizer name string to match keys
        fert_key = 'urea' # Default
        for key in self.FERT_SPECS:
            if key in recommended_fert.lower():
                fert_key = key
                break
        
        specs = self.FERT_SPECS[fert_key]
        
        # 3. Calculate Deficits
        n_deficit = max(0, target['N'] - current_nutrients['Nitrogen'])
        p_deficit = max(0, target['P'] - current_nutrients['Phosphorus'])
        k_deficit = max(0, target['K'] - current_nutrients['Potassium'])

        # 4. Calculate Quantity based on the PRIMARY nutrient of that fertilizer
        amount_kg_ha = 0
        primary_nutrient = "General"
        
        if specs['N'] > 0.2: # It's a Nitrogen fertilizer
            amount_kg_ha = n_deficit / specs['N']
            primary_nutrient = "Nitrogen"
        elif specs['P'] > 0.2: # It's a Phosphorus fertilizer
            amount_kg_ha = p_deficit / specs['P']
            primary_nutrient = "Phosphorus"
        elif specs['K'] > 0.2: # It's a Potassium fertilizer
            amount_kg_ha = k_deficit / specs['K']
            primary_nutrient = "Potassium"
        else:
            # Balanced fertilizer (average the needs)
            avg_deficit = (n_deficit + p_deficit + k_deficit) / 3
            amount_kg_ha = avg_deficit / 0.19
            primary_nutrient = "Balanced Boost"

        # Convert Hectare to Acre (1 Ha = 2.47 Acres)
        amount_kg_acre = amount_kg_ha / 2.47
        
        # Round up
        amount_kg_acre = math.ceil(amount_kg_acre)
        
        # Safety clamp (Don't recommend 0 or dangerous amounts)
        if amount_kg_acre < 10: amount_kg_acre = 10 
        if amount_kg_acre > 200: amount_kg_acre = 200 # Safety cap

        logic_msg = f"Based on {crop} needing {target['N']}N-{target['P']}P-{target['K']}K vs your soil's levels."
        
        return amount_kg_acre, logic_msg