"""
Car Price Prediction App for Hugging Face Spaces
Deploys the trained LightGBM model with quantile regression for price range prediction
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import gradio as gr
from datetime import datetime

# Configuration
ARTIFACTS_DIR = "artifacts"

# ============================================================================
# Helper Functions for Parsing Input Strings
# ============================================================================

def parse_mileage(mileage_str):
    """Parse mileage string and extract numeric value."""
    if pd.isna(mileage_str) or mileage_str == '' or mileage_str is None:
        return np.nan, np.nan
    try:
        parts = str(mileage_str).strip().split()
        if len(parts) >= 1:
            value = float(parts[0])
            unit = parts[1] if len(parts) > 1 else 'kmpl'
            return value, unit
    except:
        pass
    return np.nan, np.nan

def parse_engine(engine_str):
    """Parse engine string and extract numeric value."""
    if pd.isna(engine_str) or engine_str == '' or engine_str is None:
        return np.nan
    try:
        match = re.search(r'(\d+\.?\d*)', str(engine_str))
        if match:
            return float(match.group(1))
    except:
        pass
    return np.nan

def parse_max_power(power_str):
    """Parse max_power string and extract numeric value."""
    if pd.isna(power_str) or power_str == '' or power_str is None:
        return np.nan
    try:
        match = re.search(r'(\d+\.?\d*)', str(power_str))
        if match:
            return float(match.group(1))
    except:
        pass
    return np.nan

def parse_torque(torque_str):
    """Parse torque string and extract value, unit, and RPM."""
    if pd.isna(torque_str) or torque_str == '' or torque_str is None:
        return np.nan, np.nan, np.nan
    try:
        torque_clean = (
            str(torque_str)
            .replace(',', '')
            .replace('(', '')
            .replace(')', '')
            .replace('~', '-')
            .replace('at', '@')
            .replace('rpm', '')
            .replace('RPM', '')
            .strip()
        )
        value_match = re.search(r'^(\d+\.?\d*)', torque_clean)
        torque_value = float(value_match.group(1)) if value_match else np.nan
        unit_match = re.search(r'(Nm|nm|KGM|KGm|kgm)', torque_clean, re.IGNORECASE)
        torque_unit = unit_match.group(1).lower() if unit_match else 'nm'
        if torque_unit == 'kgm':
            torque_value = torque_value * 9.80665
        rpm_match = re.search(r'(\d+)\s*\+/-\s*\d+', torque_clean)
        if rpm_match:
            rpm_avg = float(rpm_match.group(1))
        else:
            range_match = re.search(r'(\d+)\s*-\s*(\d+)', torque_clean)
            if range_match:
                rpm_avg = (float(range_match.group(1)) + float(range_match.group(2))) / 2
            else:
                single_match = re.search(r'@\s*(\d+)', torque_clean)
                rpm_avg = float(single_match.group(1)) if single_match else np.nan
        return torque_value, torque_unit, rpm_avg
    except:
        pass
    return np.nan, np.nan, np.nan

def normalize_mileage(mileage_value, mileage_unit, fuel_type):
    """Normalize mileage to kmpl."""
    if pd.isna(mileage_value):
        return np.nan
    if mileage_unit == 'kmpl':
        return mileage_value
    elif mileage_unit == 'km/kg':
        fuel_density = {'Petrol': 0.74, 'Diesel': 0.832, 'LPG': 0.51, 'CNG': 0.615}
        density = fuel_density.get(fuel_type, 0.74)
        return mileage_value / density
    return mileage_value

# ============================================================================
# Feature Engineering
# ============================================================================

def engineer_features_from_input(row_dict, training_stats=None):
    """Engineer features from raw input data."""
    current_year = datetime.now().year
    
    # Extract make and model from name
    name = str(row_dict.get('name', '')).strip()
    name_parts = name.split() if name else []
    make = name_parts[0] if len(name_parts) > 0 else 'Unknown'
    model = name_parts[1] if len(name_parts) > 1 else 'Unknown'
    
    # Calculate age and km_per_year
    year = int(row_dict.get('year', current_year))
    age = max(1, current_year - year)
    kms_driven = int(row_dict.get('kms_driven', 0))
    km_per_year = kms_driven / age if age > 0 else 0
    
    # Parse performance attributes
    mileage_str = row_dict.get('mileage', '')
    mileage_value, mileage_unit = parse_mileage(mileage_str)
    fuel_type = row_dict.get('fuel', 'Petrol')
    mileage_kmpl = normalize_mileage(mileage_value, mileage_unit, fuel_type) if not pd.isna(mileage_value) else np.nan
    
    engine_value = parse_engine(row_dict.get('engine', ''))
    max_power_value = parse_max_power(row_dict.get('max_power', ''))
    torque_str = row_dict.get('torque', '')
    torque_value, torque_unit, rpm_avg = parse_torque(torque_str)
    
    # Frequency encodings from training statistics
    if training_stats is not None:
        make_freq_map = training_stats.get('make_freq_map', {})
        model_freq_map = training_stats.get('model_freq_map', {})
        log_price_mean = training_stats.get('log_price_mean', 0)
        model_avg_map = training_stats.get('model_avg_map', {})
        
        make_freq = make_freq_map.get(make, 0)
        model_freq = model_freq_map.get(model, 0)
        if model_avg_map and log_price_mean:
            model_avg = model_avg_map.get(model, log_price_mean)
            model_residual = model_avg - log_price_mean
        else:
            model_residual = 0
    else:
        make_freq = 0
        model_freq = 0
        model_residual = 0
    
    # Build feature dictionary
    features = {
        'age': age,
        'kms_driven': kms_driven,
        'km_per_year': km_per_year,
        'make_freq': make_freq,
        'model_freq': model_freq,
        'model_residual': model_residual,
        'mileage_value': mileage_kmpl if not pd.isna(mileage_kmpl) else np.nan,
        'engine_value': engine_value if not pd.isna(engine_value) else np.nan,
        'torque_value': torque_value if not pd.isna(torque_value) else np.nan,
        'rpm_avg': rpm_avg if not pd.isna(rpm_avg) else np.nan,
        'max_power_value': max_power_value if not pd.isna(max_power_value) else np.nan,
        'fuel': row_dict.get('fuel', 'Petrol'),
        'transmission': row_dict.get('transmission', 'Manual'),
        'owner': row_dict.get('owner', 'First Owner'),
        'make': make,
        'model': model,
        'name': name,
        'year': year,
        'seats': float(row_dict.get('seats', 5)) if row_dict.get('seats') else np.nan
    }
    
    return features

# ============================================================================
# Model Loading
# ============================================================================

def load_models():
    """Load all models and preprocessing components from artifacts directory."""
    print("Loading models from artifacts...")
    
    # Load preprocessor
    preprocessor_path = os.path.join(ARTIFACTS_DIR, 'preprocessor_lgb.joblib')
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    print("✓ Preprocessor loaded")
    
    # Load target encoder
    te_path = os.path.join(ARTIFACTS_DIR, 'target_encoder.joblib')
    if not os.path.exists(te_path):
        raise FileNotFoundError(f"Target encoder not found at {te_path}")
    target_encoder = joblib.load(te_path)
    print("✓ Target encoder loaded")
    
    # Load main LightGBM model
    model_path = os.path.join(ARTIFACTS_DIR, 'lgb_model.txt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Main model not found at {model_path}")
    model = lgb.Booster(model_file=model_path)
    print("✓ Main LightGBM model loaded")
    
    # Load quantile models
    quantile_models = {}
    for q in [0.05, 0.5, 0.95]:
        quantile_path = os.path.join(ARTIFACTS_DIR, f'lgb_quantile_{int(q*100)}.txt')
        if os.path.exists(quantile_path):
            quantile_models[q] = lgb.Booster(model_file=quantile_path)
            print(f"✓ Quantile model ({int(q*100)}th percentile) loaded")
        else:
            print(f"⚠ Quantile model ({int(q*100)}th percentile) not found")
    
    # Load feature metadata
    feature_metadata_path = os.path.join(ARTIFACTS_DIR, 'feature_metadata.joblib')
    if not os.path.exists(feature_metadata_path):
        raise FileNotFoundError(f"Feature metadata not found at {feature_metadata_path}")
    feature_metadata = joblib.load(feature_metadata_path)
    numeric_features = feature_metadata.get('numeric_features', [])
    categorical_low_card = feature_metadata.get('categorical_low_card', [])
    high_card = feature_metadata.get('high_card', [])
    print("✓ Feature metadata loaded")
    
    # Load training statistics
    training_stats_path = os.path.join(ARTIFACTS_DIR, 'training_stats.joblib')
    if not os.path.exists(training_stats_path):
        raise FileNotFoundError(f"Training statistics not found at {training_stats_path}")
    training_stats = joblib.load(training_stats_path)
    print("✓ Training statistics loaded")
    
    print("\n✅ All models loaded successfully!")
    
    return {
        'preprocessor': preprocessor,
        'target_encoder': target_encoder,
        'model': model,
        'quantile_models': quantile_models,
        'numeric_features': numeric_features,
        'categorical_low_card': categorical_low_card,
        'high_card': high_card,
        'training_stats': training_stats
    }

# ============================================================================
# Prediction Function
# ============================================================================

def predict_price_interactive(car_name, year, kms_driven, fuel, transmission, owner,
                              mileage, engine, max_power, torque, seats):
    """Predict price from user inputs."""
    try:
        # Prepare input dictionary
        input_data = {
            'name': car_name if car_name else '',
            'year': int(year) if year else datetime.now().year,
            'kms_driven': int(kms_driven) if kms_driven else 0,
            'fuel': fuel if fuel else 'Petrol',
            'transmission': transmission if transmission else 'Manual',
            'owner': owner if owner else 'First Owner',
            'mileage': mileage if mileage else '',
            'engine': engine if engine else '',
            'max_power': max_power if max_power else '',
            'torque': torque if torque else '',
            'seats': float(seats) if seats else 5.0
        }
        
        # Validate required fields
        if not car_name or not year or not kms_driven:
            return "❌ Please fill in all required fields: Car Name, Year, and Kilometers Driven"
        
        # Engineer features
        features = engineer_features_from_input(input_data, models['training_stats'])
        
        # Create DataFrame
        required_raw = ['name', 'year', 'kms_driven']
        expected_cols = list(set(
            models['numeric_features'] + 
            models['categorical_low_card'] + 
            models['high_card'] + 
            required_raw + 
            ['seats']
        ))
        
        row = {col: features.get(col, np.nan) for col in expected_cols}
        Xrow = pd.DataFrame([row])
        
        # Transform using preprocessor
        X_basic = models['preprocessor'].transform(Xrow)
        
        # Convert sparse matrix to dense array if needed
        if hasattr(X_basic, "toarray"):
            X_basic = X_basic.toarray()
        
        # Ensure high-cardinality columns exist
        for c in models['high_card']:
            if c not in Xrow.columns:
                Xrow[c] = np.nan
        
        # Apply target encoding
        X_te = models['target_encoder'].transform(Xrow[models['high_card']])
        X_te_arr = X_te.values if hasattr(X_te, "values") else np.asarray(X_te)
        
        # Combine features
        X_full = np.hstack([X_basic, X_te_arr]).astype(float)
        
        # Check if quantile models are available for price range prediction
        quantile_models = models['quantile_models']
        if quantile_models and 0.05 in quantile_models and 0.5 in quantile_models and 0.95 in quantile_models:
            # Use quantile models for prediction intervals
            pred_q = {}
            for q in [0.05, 0.5, 0.95]:
                mdl_q = quantile_models[q]
                best_it_q = getattr(mdl_q, "best_iteration", None)
                if best_it_q is not None and best_it_q > 0:
                    pred_log_q = mdl_q.predict(X_full, num_iteration=best_it_q)[0]
                else:
                    pred_log_q = mdl_q.predict(X_full)[0]
                pred_q[q] = pred_log_q
            
            # Convert from log space to original price scale
            lower_price = float(np.expm1(pred_q[0.05]))  # 5th percentile (lower bound)
            median_price = float(np.expm1(pred_q[0.5]))   # 50th percentile (median)
            upper_price = float(np.expm1(pred_q[0.95]))  # 95th percentile (upper bound)
            
            # Format output with price range
            return f"""💰 **Estimated Resale Price Range**

**Lower Estimate (5th percentile):** ₹{lower_price:,.0f}
**Median Estimate (50th percentile):** ₹{median_price:,.0f}
**Upper Estimate (95th percentile):** ₹{upper_price:,.0f}

**Recommended Price Range:** ₹{lower_price:,.0f} - ₹{upper_price:,.0f}

💡 *This 90% prediction interval provides a realistic price range. The median estimate represents the most likely price, while the range accounts for market variability. Actual market price may vary within this range.*"""
        else:
            # Fallback: Use main model with estimated uncertainty (±20% based on typical RMSE)
            main_model = models['model']
            best_it = getattr(main_model, "best_iteration", None)
            if best_it is not None and best_it > 0:
                pred_log = main_model.predict(X_full, num_iteration=best_it)[0]
            else:
                pred_log = main_model.predict(X_full)[0]
            
            # Convert from log space to original price scale
            median_price = float(np.expm1(pred_log))
            
            # Estimate range using ±20% (based on typical prediction uncertainty)
            lower_price = median_price * 0.8
            upper_price = median_price * 1.2
            
            # Format output with estimated price range
            return f"""💰 **Estimated Resale Price Range**

**Lower Estimate:** ₹{lower_price:,.0f}
**Median Estimate:** ₹{median_price:,.0f}
**Upper Estimate:** ₹{upper_price:,.0f}

**Recommended Price Range:** ₹{lower_price:,.0f} - ₹{upper_price:,.0f}

💡 *This estimated range is based on the model's prediction with ±20% uncertainty. Actual market price may vary within this range.*"""
        
    except Exception as e:
        return f"❌ Error making prediction: {str(e)}\n\nPlease check your inputs and try again."

# ============================================================================
# Load Models (Global)
# ============================================================================

print("Initializing Car Price Prediction App...")
models = load_models()

# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="Car Price Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚗 Used Car Price Predictor
        
        Enter your car specifications below to get an estimated resale price range.
        The model uses advanced machine learning techniques with quantile regression to provide 
        a realistic price range (lower, median, and upper estimates) with an R² score of 0.950.
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Required Information")
            car_name = gr.Textbox(
                label="Car Name *",
                placeholder="e.g., Maruti Swift Dzire VDI",
                info="Full name including brand and model"
            )
            year = gr.Number(
                label="Manufacturing Year *",
                value=2015,
                minimum=1990,
                maximum=datetime.now().year,
                step=1,
                info="Year the car was manufactured"
            )
            kms_driven = gr.Number(
                label="Kilometers Driven *",
                value=50000,
                minimum=0,
                step=1000,
                info="Total kilometers the car has been driven"
            )
            fuel = gr.Dropdown(
                label="Fuel Type *",
                choices=["Diesel", "Petrol", "LPG", "CNG"],
                value="Petrol",
                info="Type of fuel"
            )
            transmission = gr.Dropdown(
                label="Transmission *",
                choices=["Manual", "Automatic"],
                value="Manual",
                info="Type of transmission"
            )
            owner = gr.Dropdown(
                label="Owner *",
                choices=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
                value="First Owner",
                info="Number of previous owners"
            )
        
        with gr.Column():
            gr.Markdown("### Optional Information")
            mileage = gr.Textbox(
                label="Mileage (Optional)",
                placeholder="e.g., 23.4 kmpl or 17.3 km/kg",
                info="Fuel efficiency with unit (kmpl for Petrol/Diesel, km/kg for LPG/CNG)"
            )
            engine = gr.Textbox(
                label="Engine (Optional)",
                placeholder="e.g., 1248 CC",
                info="Engine displacement in CC"
            )
            max_power = gr.Textbox(
                label="Max Power (Optional)",
                placeholder="e.g., 74 bhp",
                info="Maximum power output in bhp"
            )
            torque = gr.Textbox(
                label="Torque (Optional)",
                placeholder="e.g., 190Nm@ 2000rpm",
                info="Torque specification with RPM"
            )
            seats = gr.Number(
                label="Number of Seats (Optional)",
                value=5,
                minimum=2,
                maximum=10,
                step=1,
                info="Number of seats in the car"
            )
    
    predict_btn = gr.Button("🚀 Predict Price", variant="primary", size="lg")
    
    output = gr.Markdown(
        label="Prediction Result",
        value="Enter car details and click 'Predict Price' to get an estimate."
    )
    
    # Example inputs
    gr.Markdown("### 💡 Example Inputs")
    gr.Examples(
        examples=[
            ["Maruti Swift Dzire VDI", 2014, 145500, "Diesel", "Manual", "First Owner", "23.4 kmpl", "1248 CC", "74 bhp", "190Nm@ 2000rpm", 5],
            ["Honda City 2017-2020 EXi", 2018, 35000, "Petrol", "Automatic", "First Owner", "17.8 kmpl", "1498 CC", "121 bhp", "145Nm@ 4300rpm", 5],
            ["Hyundai i20 Sportz", 2016, 80000, "Petrol", "Manual", "Second Owner", "18.5 kmpl", "1197 CC", "83 bhp", "115Nm@ 4000rpm", 5],
        ],
        inputs=[car_name, year, kms_driven, fuel, transmission, owner, mileage, engine, max_power, torque, seats],
        label="Click any example to auto-fill the form"
    )
    
    # Connect prediction function
    predict_btn.click(
        fn=predict_price_interactive,
        inputs=[car_name, year, kms_driven, fuel, transmission, owner, mileage, engine, max_power, torque, seats],
        outputs=output
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Model Performance:**
        - Test R²: 0.950
        - RMSE: ₹102,484
        - MAE: ₹57,977
        - 90% Prediction Interval Coverage: ~90%
        
        *Note: The price range (lower, median, upper estimates) provides a realistic interval based on quantile regression. 
        The median represents the most likely price, while the range accounts for market variability. 
        Actual market prices may vary within this range.*
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

