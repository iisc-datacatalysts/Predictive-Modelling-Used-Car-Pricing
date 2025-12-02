# Deployment Guide for Hugging Face Spaces

This guide will help you deploy the Car Price Prediction app to Hugging Face Spaces.

## Prerequisites

1. A Hugging Face account (sign up at https://huggingface.co/)
2. All model artifacts saved in the `artifacts/` folder
3. Git installed on your local machine

## Step 1: Prepare Your Files

Ensure you have the following files in your project directory:

```
car-price-prediction/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # Project README (optional)
├── README_HF.md             # Hugging Face Spaces README
└── artifacts/                # Model artifacts folder
    ├── preprocessor_lgb.joblib
    ├── target_encoder.joblib
    ├── lgb_model.txt
    ├── lgb_quantile_5.txt
    ├── lgb_quantile_50.txt
    ├── lgb_quantile_95.txt
    ├── feature_metadata.joblib
    └── training_stats.joblib
```

## Step 2: Create a New Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in the details:
   - **Space name**: `car-price-predictor` (or your preferred name)
   - **SDK**: Select "Gradio"
   - **Visibility**: Public or Private (your choice)
4. Click "Create Space"

## Step 3: Upload Files to Hugging Face

### Option A: Using Git (Recommended)

1. **Initialize Git repository** (if not already done):
   ```bash
   cd car-price-prediction
   git init
   ```

2. **Add Hugging Face remote**:
   ```bash
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```
   Replace `YOUR_USERNAME` and `YOUR_SPACE_NAME` with your actual values.

3. **Add and commit files**:
   ```bash
   git add app.py requirements.txt README_HF.md artifacts/
   git commit -m "Initial commit: Car Price Predictor app"
   ```

4. **Push to Hugging Face**:
   ```bash
   git push origin main
   ```

### Option B: Using Web Interface

1. Go to your Space page on Hugging Face
2. Click "Files and versions" tab
3. Click "Add file" → "Upload files"
4. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md` (rename `README_HF.md` to `README.md` for Hugging Face)
   - All files from `artifacts/` folder

## Step 4: Verify Deployment

1. Go to your Space page
2. The app should automatically build and deploy
3. Wait for the build to complete (usually 2-5 minutes)
4. Once deployed, you'll see "Running" status
5. Test the app with the example inputs

## Step 5: Update README (Optional)

Hugging Face Spaces uses `README.md` for the Space description. You can:

1. Rename `README_HF.md` to `README.md`, or
2. Copy the content from `README_HF.md` to `README.md` in your Space

## Troubleshooting

### Build Fails

- Check that all required files are uploaded
- Verify `requirements.txt` has correct package versions
- Check the build logs in the "Logs" tab

### Models Not Loading

- Ensure all 8 files are in the `artifacts/` folder
- Check file names match exactly (case-sensitive)
- Verify file sizes are reasonable (not 0 bytes)

### App Crashes on Prediction

- Check the logs for error messages
- Verify all model files are complete
- Ensure input validation is working correctly

## File Size Considerations

Model files can be large. Hugging Face Spaces has:
- **Free tier**: 50GB storage
- **Pro tier**: 100GB storage

If you exceed limits:
- Consider using Git LFS for large files
- Or compress models (though this may affect loading time)

## Updating the App

To update your deployed app:

1. Make changes to `app.py` locally
2. Commit and push:
   ```bash
   git add app.py
   git commit -m "Update app"
   git push origin main
   ```
3. Hugging Face will automatically rebuild and redeploy

## Sharing Your App

Once deployed, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

Share this URL with others to let them use your car price predictor!

## Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [Git LFS for Large Files](https://git-lfs.github.com/)

