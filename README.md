# 🩺 Diabetic Retinopathy Detection (Deep Learning)

A **deep learning-based web application** that detects diabetic retinopathy severity from retinal fundus images using **EfficientNet-B0**, with **probability scores and Grad-CAM visual explanations**.

---

## 🌐 Live Demo

👉 https://diabetic-retinopathy-detection-axv7newj2sw24tanqekc2a.streamlit.app

---

## 📌 Features

* 🔍 **5-Class Classification**

  * No DR
  * Mild
  * Moderate
  * Severe
  * Proliferative DR

* 📊 **Class Probability Visualization**

  * Displays probability scores for all classes
  * Helps in understanding model confidence

* 🔥 **Grad-CAM Explainability**

  * Highlights important regions in retinal image
  * Improves model transparency

* ⚡ **Real-Time Prediction**

  * Upload image → instant diagnosis

* 🎯 **Confidence Score Display**

  * Shows prediction certainty percentage

---

## 📸 Screenshots

### 🖥️ Prediction Interface

*(Add your uploaded image link here from GitHub)*

### 📊 Class Probabilities Output

*(Add screenshot showing probability bars)*

### 🔥 Grad-CAM Visualization

*(Add screenshot showing heatmap output)*

---

## ⚙️ How It Works

1. User uploads a retinal fundus image
2. Image is preprocessed (resize, normalization)
3. Passed through **EfficientNet-B0 model**
4. Model outputs probabilities for all 5 classes
5. Highest probability class selected as prediction
6. Grad-CAM generates activation heatmap
7. Results displayed with confidence score

---

## 🧪 Dataset

* **APTOS 2019 Blindness Detection (Kaggle)**
* Includes labeled retinal fundus images
* Preprocessing includes:

  * Resizing
  * Normalization
  * Data augmentation

---

## 📊 Model Performance

| Metric              | Score |
| ------------------- | ----- |
| Validation Accuracy | XX%   |
| QWK Score           | 0.XX  |

---

## 🛠️ Tech Stack

* **Frontend/UI**: Streamlit
* **Backend**: Python
* **Deep Learning**: PyTorch, torchvision, timm
* **Image Processing**: OpenCV, Pillow
* **Explainability**: Grad-CAM

---

## 🧩 Project Structure

```
├── app.py                      # Streamlit application
├── best_model.pth              # Trained model weights
├── diabetic-retinopathy.ipynb  # Model training notebook
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/Sufiyan78666/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Future Improvements

* 📈 Add training accuracy/loss graphs
* 🌐 Convert into REST API (Flask/FastAPI)
* 📱 Mobile app integration
* 🏥 Clinical dataset validation

---

## ⚠️ Limitations

* Model trained on limited dataset
* Not clinically validated
* Performance depends on image quality

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only** and **not intended for clinical use**.

---

## 💡 Key Highlights

* End-to-end deep learning pipeline
* Real-time inference system
* Explainable AI using Grad-CAM
* Cloud deployment using Streamlit

---

## 👨‍💻 Author

**Sufiyan Khan**

* GitHub: https://github.com/Sufiyan78666


---

⭐ If you like this project, give it a star on GitHub!
