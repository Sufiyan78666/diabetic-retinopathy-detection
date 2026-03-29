# Diabetic Retinopathy Detection

A deep learning web app that detects diabetic retinopathy severity
from retinal fundus images using EfficientNet-B0.

## Demo
[Live App](https://huggingface.co/spaces/sufiyan78666/diabetic-retinopathy-detection)

## About
- **Model**: EfficientNet-B0 (transfer learning)
- **Dataset**: APTOS 2019 Blindness Detection (Kaggle)
- **Classes**: No DR, Mild, Moderate, Severe, Proliferative DR
- **Explainability**: Grad-CAM heatmaps

## Results
| Metric | Score |
|--------|-------|
| Validation Accuracy | XX% |
| QWK Score | 0.XX |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
├── app.py              # Streamlit web app
├── diabetic-retinopathy.ipynb      # Training notebook
├── best_model.pth      # Trained model weights
└── requirements.txt    # Dependencies
```
