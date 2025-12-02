---
title: Used Car Price Predictor
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🚗 Used Car Price Predictor

A machine learning-powered web application for predicting used car resale prices in the Indian market. The model uses advanced LightGBM with quantile regression to provide realistic price ranges (lower, median, and upper estimates).

## Features

- **Price Range Prediction**: Provides 90% prediction intervals (5th, 50th, 95th percentiles)
- **High Accuracy**: R² score of 0.950 on test data
- **User-Friendly Interface**: Simple form-based input with example values
- **Comprehensive Inputs**: Supports both required and optional car specifications

## Model Performance

- **Test R²**: 0.950
- **RMSE**: ₹102,484
- **MAE**: ₹57,977
- **90% Prediction Interval Coverage**: ~90%

## Usage

1. Fill in the required fields:
   - Car Name (e.g., "Maruti Swift Dzire VDI")
   - Manufacturing Year
   - Kilometers Driven
   - Fuel Type
   - Transmission Type
   - Owner Type

2. Optionally fill in additional specifications:
   - Mileage
   - Engine displacement
   - Max Power
   - Torque
   - Number of Seats

3. Click "🚀 Predict Price" to get an estimated price range

## Model Details

The model uses:
- **LightGBM** with target encoding for high-cardinality features
- **KNN Imputation** for missing value handling
- **Polynomial Features** for capturing feature interactions
- **Quantile Regression** for uncertainty estimation

## Files Required

Make sure the following files are in the `artifacts/` directory:

- `preprocessor_lgb.joblib` - Preprocessing pipeline
- `target_encoder.joblib` - Target encoder
- `lgb_model.txt` - Main LightGBM model
- `lgb_quantile_5.txt` - 5th percentile quantile model
- `lgb_quantile_50.txt` - 50th percentile quantile model
- `lgb_quantile_95.txt` - 95th percentile quantile model
- `feature_metadata.joblib` - Feature lists and metadata
- `training_stats.joblib` - Training statistics for frequency encoding

## Deployment

This app is deployed on Hugging Face Spaces. To deploy your own version:

1. Upload all model artifacts to the `artifacts/` folder in your Hugging Face Space
2. Ensure `app.py` and `requirements.txt` are in the root directory
3. The app will automatically load models on startup

## License

MIT License - See LICENSE file for details

