# Project Title : Predictive Modelling for Used Car Pricing

A comprehensive machine learning project for predicting used car prices using various regression models and advanced feature engineering techniques.
## 👥 Team Members:  
- Vimalraj Kanagaraj (vimalrajk@iisc.ac.in) - SR No: 13-19-02-19-52-25-1-26550 
- Anfaal Obaid Waafy (anfaalwaafy@iisc.ac.in) - SR No: 13-19-02-19-52-25-1-26283
- Manikanda Sakthi Subramaniam (manikandasa1@iisc.ac.in) - SR No: 13-19-02-19-52-25-1-26552
- Abhilasha Kawle (abhilashak@iisc.ac.in) - SR No: 13-19-02-19-52-25-1-26323

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Notebook](#running-the-notebook)
  - [Making Predictions](#making-predictions)
  - [Interactive Price Prediction Dashboard](#interactive-price-prediction-dashboard)
- [Methodology](#methodology)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)

# 🧩 Problem Statement: Predicting Market Price of Used Cars

## 📍 Background
- **Challenge:** Buyers and sellers struggle to determine the fair market value of used cars.
- **Cause:** Lack of accurate, data-driven estimates.
  - Manual inspection and estimation are inconsistent and inaccurate.
  - Different platforms show wide price variance.

## 🎯 Importance
- **For Customers:** Builds transparency and trust.
- **For Businesses (Marketplaces, Dealerships):**
  - Enables automated, reliable price prediction for informed decisions.
  - Reduces negotiation gaps, increasing sales conversions.

## 🛠️ Project Objectives
- Build a data-driven model to predict the market price of used cars based on their features.
- Ensure model interpretability and deliver a deployable pipeline.

## 🧠 Data Science Approach
- Leverage historical listings and feature engineering to learn price drivers.

## 🎯 Solution Overview

This project implements an end-to-end machine learning pipeline for predicting used car prices in the Indian market. The solution includes:

> **🚀 Try the Interactive Dashboard**: 
> - **🌐 Live Demo**: [**Access the deployed dashboard on Hugging Face Spaces**](https://huggingface.co/spaces/vimalkanagaraj/used-car-price-prediction)
> - **📓 Notebook**: Run the notebook and launch the Gradio interface (Cells 80-89) to get a shareable public URL for the price prediction dashboard. The URL will be displayed in the notebook output when you run Cell 89.

- **Comprehensive Data Cleaning**: Handling missing values, normalizing units, and extracting features from mixed-format columns
- **Enhanced Preprocessing**: 
  - **KNN Imputation**: Using K-Nearest Neighbors imputation (n_neighbors=5) for sophisticated missing value handling
  - **Polynomial Features**: Generating interaction features (degree=2, interaction_only=True) to capture feature relationships
  - Standard scaling for numerical features
- **Advanced Feature Engineering**: Creating derived features like car age, km/year, frequency encodings, and residual encodings
- **Multiple Model Evaluation**: Comparing baseline and advanced models including Ridge, Random Forest, XGBoost, CatBoost, and LightGBM
- **Target Encoding**: Using target encoding for high-cardinality categorical features
- **Quantile Regression**: Providing prediction intervals for uncertainty estimation (5th, 50th, 95th percentiles)
- **Model Interpretability**: SHAP analysis for feature importance
- **Error Analysis**: Systematic analysis of prediction patterns (underprediction for luxury cars, overprediction for rare fuel types)
- **Production-Ready Inference**: Serialized models and inference function for deployment
- **Google Drive Integration**: Persistent storage of model artifacts in Google Drive for Colab sessions
- **Interactive Dashboard**: Gradio-based web interface for real-time price range predictions

## 📊 Dataset

The dataset contains information about **8,128 used cars** with the following attributes:

### Features

**Car Identification & Basic Info:**
- `name`: Full car name (combines brand and model)
- `year`: Manufacturing year
- `selling_price`: Target variable - the price at which the car was sold

**Car Usage History:**
- `km_driven`: Total kilometers driven
- `owner`: Ownership history (First Owner, Second Owner, etc.)
- `seller_type`: Type of seller (Individual, Dealer, Trustmark Dealer)

**Car Specifications:**
- `fuel`: Fuel type (Diesel, Petrol, LPG, CNG)
- `transmission`: Transmission type (Manual, Automatic)
- `mileage`: Fuel efficiency (with units: kmpl or km/kg)
- `engine`: Engine displacement (in CC)
- `max_power`: Maximum power output (in bhp)
- `torque`: Torque specification (in Nm or kgm, with RPM range)
- `seats`: Number of seats

**Dataset Source:** The dataset is loaded from the IISC Data Catalysts GitHub repository. The original dataset is maintained by Kaggle team - [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho). This has been downloaded from Kaggle and uploaded into GitHub for ease of access.

## 🔧 Features

### Data Preprocessing
- **KNN Imputation**: Uses K-Nearest Neighbors (n_neighbors=5) to impute missing values based on similar observations (more sophisticated than median imputation)
- Missing value imputation (constant for categorical)
- Unit normalization (km/kg → kmpl, kgm → Nm)
- String parsing for mixed-format columns (mileage, engine, max_power, torque)
- RPM extraction from complex torque strings
- **Polynomial Features**: Generates interaction features (degree=2, interaction_only=True) to capture multiplicative relationships between features
- Standard scaling for numerical features

### Feature Engineering
- **Derived Features**: Car age, kilometers per year, abnormal usage flags
- **Frequency Encoding**: Normalized frequency of make and model occurrences
- **Residual Encoding**: Model-specific price deviations from overall mean
- **Log Transformation**: Applied to target variable to handle right-skewed distribution

### Model Training
- Stratified train-test split (85% train, 15% test)
- Cross-validation with honest target encoding
- Early stopping for gradient boosting models
- Hyperparameter tuning for optimal performance

## 📁 Project Structure

```
Predictive-Modelling-Used-Car-Pricing/
├── README.md                                       # Main project documentation
├── LICENSE                                         # License file
├── requirements.txt                                # Python dependencies for the entire project
├── data/
│   └── CarData.csv                                 # Dataset (can also be loaded from GitHub)
├── code/
│   ├── notebook/
│   │   ├── DSPCourse_Project_UsedCarPricePrediction.ipynb  # Main Jupyter notebook
│   │   └── README.md                              # Notebook-specific documentation
│   └── interactive-dashboard-ui/
│       ├── app.py                                 # Standalone Gradio app for Hugging Face deployment
│       ├── requirements.txt                       # Dashboard-specific dependencies
│       ├── README.md                              # Dashboard documentation
│       ├── DEPLOYMENT_GUIDE.md                    # Deployment instructions
│       ├── QUICK_START.md                         # Quick deployment guide
│       ├── LICENSE                                # License file
│       └── artifacts/                             # Saved models (generated after running notebook)
│           ├── preprocessor_lgb.joblib            # Preprocessing pipeline
│           ├── target_encoder.joblib              # Target encoder
│           ├── lgb_model.txt                      # Main LightGBM model
│           ├── lgb_quantile_5.txt                 # 5th percentile quantile model
│           ├── lgb_quantile_50.txt                # 50th percentile quantile model
│           ├── lgb_quantile_95.txt                # 95th percentile quantile model
│           ├── feature_metadata.joblib            # Feature metadata
│           └── training_stats.joblib              # Training statistics
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DSP
   ```

2. **Install required packages:**
   
   The notebook will automatically install packages in the first cell, or you can install them manually:
   
   ```bash
   pip install shap lightgbm category_encoders catboost scikit-learn pandas numpy matplotlib seaborn gradio
   ```
   
   Note: Gradio is installed automatically when you run the interactive dashboard cells (Cell 80).

3. **Open the notebook:**
   ```bash
   jupyter notebook DSPCourse_Project_UsedCarPricePrediction.ipynb
   ```

## 💻 Usage

### Running the Notebook

1. Open the notebook in Jupyter
2. Run all cells sequentially (Cell → Run All)
3. The notebook will:
   - Load data from GitHub
   - Perform data cleaning and feature engineering
   - Train multiple models
   - Evaluate performance
   - Save model artifacts to Google Drive (`/content/drive/MyDrive/artifacts`) in Colab, or local `artifacts/` directory otherwise
   - Perform error analysis on prediction patterns

### Making Predictions

After training, you can use the `predict_single()` function to predict prices for new cars:

```python
# Example usage
example_car = {
    'name': 'Maruti Swift Dzire VDI',
    'year': 2014,
    'kms_driven': 145500,
    'fuel': 'Diesel',
    'transmission': 'Manual',
    'owner': 'First Owner',
    'mileage': '23.4 kmpl',
    'engine': '1248 CC',
    'max_power': '74 bhp',
    'torque': '190Nm@ 2000rpm',
    'seats': 5.0
}

predicted_price = predict_single(example_car)
print(f"Predicted Price: ₹{predicted_price:,.0f}")
```

**Note:** `seller_type` is present in the dataset but is not used as a feature in the model. The model uses `fuel`, `transmission`, and `owner` as categorical features. You can omit `seller_type` from the input dictionary.

### Interactive Price Prediction Dashboard

**🌐 Live Dashboard**: [**Try the deployed version on Hugging Face Spaces**](https://huggingface.co/spaces/vimalkanagaraj/used-car-price-prediction)

The notebook includes an interactive Gradio web interface for easy price predictions. The dashboard is embedded directly in the notebook and provides a user-friendly way to estimate car prices. A permanent version is also deployed on Hugging Face Spaces for easy access.

**To use the dashboard:**

1. **Run all previous cells** (especially model training cells) to ensure the model is trained
2. **Run Cells 80-89** to launch the Gradio interface:
   - Cell 80: Setup Google Drive for loading artifacts (in Colab)
   - Cell 81: Load models from artifacts directory
   - Cell 82-88: Define helper functions for parsing, feature engineering, and prediction
   - Cell 89: Create and launch the Gradio interface

**Dashboard Features:**
- **Two-column layout**: Required fields on the left, optional fields on the right
- **Input validation**: Checks required fields before prediction
- **Example inputs**: 3 pre-filled examples you can click to auto-fill the form
- **Price Range Predictions**: Provides lower estimate (5th percentile), median estimate (50th percentile), and upper estimate (95th percentile) using quantile regression models
- **90% Prediction Interval**: Shows realistic price range accounting for market variability
- **Real-time predictions**: Instant price range estimates with formatted display
- **Public URL**: Automatically generates a shareable link (works great in Google Colab)
- **Embedded interface**: Displays directly in the notebook output

**Accessing the Dashboard:**

- **🌐 Permanent Hosted Version**: [**https://huggingface.co/spaces/vimalkanagaraj/used-car-price-prediction**](https://huggingface.co/spaces/vimalkanagaraj/used-car-price-prediction) - Always available, no expiration
- **📓 Notebook Version**: When you run Cell 89, Gradio automatically generates a public URL that will be displayed in the notebook output. The URL format is typically:
  - **Public URL**: `https://xxxxx.gradio.live` (shareable, expires in 1 week)
  - **Local URL**: `http://127.0.0.1:7860` (works when running locally in Jupyter)

**Example Output:**
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live

This share link expires in 1 week. For free permanent hosting and GPU upgrades, 
run `gradio deploy` from the terminal to deploy to Hugging Face Spaces.
```

**Important Notes:**
- **🌐 Permanent Hosted Version**: The dashboard is permanently hosted at [Hugging Face Spaces](https://huggingface.co/spaces/vimalkanagaraj/used-car-price-prediction) - no expiration, always accessible
- **📓 Notebook Version**: The public URL is automatically generated and shareable with anyone
- Free Gradio URLs from the notebook expire after 1 week
- A new URL is generated each time you run the dashboard in the notebook
- The interface is also embedded directly in the notebook output
- The dashboard uses quantile regression models to provide price ranges (lower, median, upper estimates)

**Required Fields:**
- Car Name (e.g., "Maruti Swift Dzire VDI")
- Manufacturing Year
- Kilometers Driven
- Fuel Type (Diesel, Petrol, LPG, CNG)
- Transmission (Manual, Automatic)
- Owner (First Owner, Second Owner, etc.)

**Optional Fields:**
- Mileage (format: "23.4 kmpl" or "17.3 km/kg")
- Engine (format: "1248 CC")
- Max Power (format: "74 bhp")
- Torque (format: "190Nm@ 2000rpm")
- Number of Seats

**Note:** The dashboard uses the trained model variables directly from the notebook's namespace, so make sure all model training cells have been executed first.


## 📈 Methodology

### Model Artifact Storage

The notebook includes Google Drive integration for persistent storage of model artifacts:

- **Cell 69**: Setup Google Drive mounting and define artifacts directory (`/content/drive/MyDrive/artifacts`)
- **Cell 70**: Main artifact saving (preprocessor, target encoder, models, quantile models)
- **Cell 71**: Additional metadata saving (feature metadata, training statistics)
- **Cell 72**: Copy existing local artifacts to Google Drive (if needed)

This ensures model artifacts persist across Colab sessions. When running locally, artifacts are saved to the local `artifacts/` directory.

### Error Analysis

The notebook includes comprehensive error analysis (Cell 78-79) that identifies systematic prediction patterns:

1. **Underprediction for Luxury Cars**:
   - Average error: -2.69% for luxury/high-end brands
   - 59.7% of luxury cars are underpriced
   - Justified by limited training data and brand premiums not captured by features

2. **Overprediction for Rare Fuel Types**:
   - Average overprediction: 14.58% for LPG/CNG cars
   - 53.3% of LPG/CNG cars are overpriced
   - Justified by very limited training data (only 1.2% of dataset) and conversion-related depreciation

This analysis provides insights for model improvement and helps users understand prediction limitations.

### Data Cleaning Pipeline

1. **Name Parsing**: Extract make (brand) and model from car names
2. **Mileage Normalization**: Convert km/kg to kmpl using fuel density ratios
3. **Torque Processing**: Extract torque values and normalize units (kgm → Nm)
4. **RPM Extraction**: Parse RPM information from various formats (ranges, +/- notation)

### Enhanced Preprocessing Pipeline

1. **KNN Imputation**: Uses K-Nearest Neighbors (n_neighbors=5) to impute missing values. This is more sophisticated than median imputation as it considers relationships between features to provide more accurate missing value estimates.
2. **Standard Scaling**: Normalizes features to have zero mean and unit variance
3. **Polynomial Features**: Creates interaction features (degree=2, interaction_only=True) to capture multiplicative relationships between features (e.g., age × kms_driven, age × max_power_value). The `interaction_only=True` parameter means only interaction terms are created, not squared terms.

### Feature Engineering

1. **Temporal Features**: Calculate car age from manufacturing year
2. **Usage Features**: Compute kilometers per year and flag abnormal usage
3. **Encoding Features**: 
   - Frequency encoding for make/model popularity
   - Residual encoding for model-specific price patterns
4. **Target Transformation**: Apply log1p transformation to handle skewness

### Model Training Strategy

1. **Baseline Models**: Start with simple linear models (Ridge) with enhanced preprocessing (KNN imputation + polynomial features)
2. **Tree-Based Models**: Evaluate Random Forest, XGBoost, CatBoost, LightGBM
3. **Ensemble Methods**: Stack multiple models for improved performance
4. **Advanced Techniques**: 
   - Target encoding for high-cardinality features (make, model)
   - Quantile regression for uncertainty estimation
   - Cross-validation with honest target encoding (out-of-fold predictions)
   - Early stopping for gradient boosting models

## 🤖 Models Evaluated

| Model | Description | Key Features |
|-------|-------------|--------------|
| **Ridge Regression** | Linear baseline model | L2 regularization, fast training |
| **Random Forest** | Ensemble of decision trees | Non-linear relationships, feature interactions |
| **XGBoost** | Gradient boosting framework | Regularization, subsampling |
| **CatBoost** | Categorical boosting | Native categorical handling |
| **LightGBM** | Gradient boosting framework | Fast training, low memory usage |
| **Stacking Ensemble** | Meta-learner combining models | Combines predictions from multiple models |
| **LightGBM + Target Encoding** | Advanced model with encoding | Best performance, handles high-cardinality features |

## 📊 Results

### Model Performance

The LightGBM model with target encoding achieved the best performance:

- **Training R²**: 0.984
- **Test R²**: 0.950
- **RMSE**: ₹102,484
- **MAE**: ₹57,977

### Cross-Validation Results

5-fold cross-validation R² scores:
- **Ridge Regression**: 0.84 - 0.94 (varies by fold)
- **CatBoost**: 0.94 - 0.95
- **LightGBM**: 0.94 - 0.95
- **XGBoost**: 0.93 - 0.95

### Key Findings

1. **Strong Predictors**:
   - Car age (negative correlation: -0.71)
   - Kilometers driven (negative correlation: -0.25)
   - Max power and engine size (positive correlations)

2. **Categorical Insights**:
   - Diesel cars command higher prices than Petrol/LPG/CNG
   - Automatic transmission increases price
   - First owner cars are most valuable

3. **Prediction Intervals**:
   - 90% prediction interval coverage: ~90%
   - Quantile regression provides uncertainty estimates (5th, 50th, 95th percentiles)
   - Price range predictions include lower bound (5th percentile), median (50th percentile), and upper bound (95th percentile)

4. **Prediction Pattern Analysis**:
   - **Underprediction for luxury/high-end cars**: Model tends to underestimate prices for premium brands (average error: -2.69%) due to limited training data and brand value premiums not fully captured
   - **Overprediction for rare fuel types (LPG/CNG)**: Model tends to overestimate prices for LPG/CNG cars (average overprediction: 14.58%) due to limited training data (only 0.7% CNG, 0.5% LPG) and conversion-related depreciation factors

## 🛠 Technologies Used

- **Python 3.10+**
- **Data Science Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning
- **Advanced ML Libraries**:
  - `lightgbm` - Gradient boosting
  - `xgboost` - Gradient boosting
  - `catboost` - Categorical boosting
  - `category_encoders` - Target encoding
- **Visualization**:
  - `matplotlib` - Plotting
  - `seaborn` - Statistical visualization
  - `shap` - Model interpretability
- **Interactive UI**:
  - `gradio` - Web interface for model predictions
- **Utilities**:
  - `joblib` - Model serialization

## 🔍 Key Insights

1. **Data Quality**: The dataset contains missing values in performance attributes (~221 missing in mileage, engine, max_power, torque, seats), which are handled through KNN imputation (n_neighbors=5) - a more sophisticated approach than median imputation

2. **Feature Importance**: Car age is the strongest predictor, followed by kilometers driven and engine specifications

3. **Model Selection**: LightGBM with target encoding outperforms other models, balancing accuracy and training time

4. **Target Transformation**: Log transformation is crucial for handling the right-skewed price distribution

5. **High-Cardinality Features**: Target encoding effectively handles the 2,058 unique car names and models

6. **Prediction Patterns**:
   - **Luxury Cars**: Model shows systematic underprediction (-2.69% average error) due to limited training data and brand premiums not fully captured by features
   - **Rare Fuel Types**: LPG/CNG cars show overprediction (14.58% average) due to limited training examples (only 1.2% of dataset) and conversion-related depreciation factors
   - These patterns are justified by data imbalance and feature limitations, providing insights for model improvement

7. **Google Drive Integration**: Model artifacts are automatically saved to Google Drive (`/content/drive/MyDrive/artifacts`) when running in Colab, ensuring persistence across sessions

## 🔮 Future Improvements

- **Enhanced Name Parsing**: Use regex patterns or fuzzy matching against curated make/model databases for more accurate extraction
- **Leakage Testing**: Evaluate models with and without potentially leaked features to ensure robustness
- **Calibrated Intervals**: Implement conformalized quantile regression for better-calibrated uncertainty estimates
- **Additional Features**: Explore brand reputation scores, depreciation curves, market segment indicators
- **Polynomial Feature Selection**: Consider feature selection techniques to reduce dimensionality after polynomial feature generation
- **KNN Imputation Tuning**: Experiment with different values of n_neighbors for KNN imputation to optimize missing value handling
- **Model Optimization**: Hyperparameter tuning with Optuna or similar tools
- **Testing**: Add unit tests for preprocessing pipeline and inference function before production deployment
- **Computational Efficiency**: Monitor training time with polynomial features; consider feature selection if dimensionality becomes too high
- **Deployment**: Deploy to Hugging Face Spaces (deployment files already created: `app.py`, `requirements.txt`, `README_HF.md`, `DEPLOYMENT_GUIDE.md`)
- **Dashboard Enhancements**: Add feature importance visualization and comparison with similar cars
- **Error Correction**: Collect more training data for luxury cars and rare fuel types to reduce systematic biases

## 📝 Notes

- The dataset is automatically loaded from the IISC Data Catalysts GitHub repository
- Python 3.10+ is recommended for running this notebook
- Required packages are installed in the first code cell of the notebook
- Gradio is automatically installed when running the interactive dashboard cells (Cell 80)
- Model artifacts are saved to Google Drive (`/content/drive/MyDrive/artifacts`) in Colab, or local `artifacts/` directory otherwise
- The notebook includes comprehensive documentation and comments
- The enhanced preprocessing pipeline uses KNN imputation and polynomial interaction features
- The interactive Gradio dashboard provides price range predictions (lower, median, upper estimates) using quantile regression
- The dashboard is embedded in the notebook and works seamlessly in both Jupyter and Google Colab
- Error analysis section (Cell 78-79) provides insights into prediction patterns and systematic biases
- Results may vary slightly due to random seeds and data splits
- For deployment to Hugging Face Spaces, use the provided `app.py` and follow `DEPLOYMENT_GUIDE.md`

## Dataset: [Kaggale](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

## 📄 License

This project is part of a course assignment. Please refer to the course guidelines for usage terms.

## 🙏 Acknowledgments

- Dataset provided via GitHub repository
- Course instructors and materials
- Open-source ML community for excellent libraries

---

**Note**: This project is for educational purposes. Model performance may vary with different datasets or market conditions.

