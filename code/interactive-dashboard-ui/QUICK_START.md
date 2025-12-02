# Quick Start Guide

## Files Created for Hugging Face Deployment

✅ **app.py** - Main application file with Gradio interface
✅ **requirements.txt** - Python dependencies
✅ **README_HF.md** - Hugging Face Spaces README
✅ **DEPLOYMENT_GUIDE.md** - Step-by-step deployment instructions
✅ **.gitignore** - Git ignore file

## Your Artifacts Folder

All required model files are present in the `artifacts/` folder:
- ✅ preprocessor_lgb.joblib
- ✅ target_encoder.joblib
- ✅ lgb_model.txt
- ✅ lgb_quantile_5.txt
- ✅ lgb_quantile_50.txt
- ✅ lgb_quantile_95.txt
- ✅ feature_metadata.joblib
- ✅ training_stats.joblib

## Quick Deployment Steps

1. **Create Hugging Face Space**:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select "Gradio" SDK
   - Name your space (e.g., "car-price-predictor")

2. **Upload Files**:
   - Upload `app.py` to root
   - Upload `requirements.txt` to root
   - Upload `README.md` (copy content from `README_HF.md`)
   - Upload entire `artifacts/` folder

3. **Wait for Build**:
   - Hugging Face will automatically build your app
   - Check "Logs" tab if there are any issues
   - App will be live once build completes!

## Testing Locally (Optional)

Before deploying, you can test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will start at `http://localhost:7860`

## What the App Does

- Takes car specifications as input
- Uses trained LightGBM model with quantile regression
- Returns price range (lower, median, upper estimates)
- Provides 90% prediction intervals

## Need Help?

See `DEPLOYMENT_GUIDE.md` for detailed instructions and troubleshooting.

