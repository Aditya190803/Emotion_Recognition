# 😊 Facial Emotion Recognition — Production Ready

End-to-end facial emotion recognition system with automatic dataset download, deep learning model training, and a modern Streamlit web UI — designed for both **local development** and **Streamlit Community Cloud** deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58%2B-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## ✨ Features

- **🌐 Automatic Dataset Download** — One-click FER2013 download from the sidebar
- **🏋️ In-App Model Training** — Train new CNN models directly from the web UI (runs in background)
- **📁 Model Selection** — Switch between multiple `.keras` checkpoints from the sidebar
- **📷 Real-Time Webcam Detection** — Live camera feed with face bounding boxes and emotion overlays *(local only)*
- **🖼️ Image Upload** — Drag & drop any image for instant prediction with confidence bars
- **📊 Model Dashboard** — Dataset distribution charts, training history plots, and model architecture
- **☁️ Streamlit Cloud Ready** — Deploys to Streamlit Community Cloud with zero config

## 🚀 Quick Start

```bash
# 1. Create virtual environment (Python 3.11)
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the Streamlit app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
.
├── app.py                         # Main Streamlit web application
├── download_dataset.py            # FER2013 dataset downloader
├── train_model.py                 # CNN model training
├── realtime_prediction.py         # Standalone webcam script
├── capture_data.py                # Manual webcam data collection
├── requirements.txt               # Python dependencies
├── packages.txt                   # System deps for Streamlit Cloud (OpenCV)
├── .env.example                   # Configuration template
├── .gitignore                     # Git ignore rules
├── .streamlit/
│   └── config.toml                # Streamlit theme & server config
├── venv/                          # Virtual environment
├── dataset/                       # Dataset folder (generated after download)
│   ├── train/
│   ├── validation/
│   └── test/
├── emotion_model_*.keras          # Trained models (generated)
├── training_history_*.json        # Training metrics (generated)
└── README.md
```

---

## 🛠️ Installation

### Prerequisites
- Python **3.11** (required for TensorFlow compatibility)
- pip
- Webcam *(for local real-time detection only)*

### Setup

```bash
cd Emotion_Recognition

# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Sidebar Workflow

The app is controlled entirely from the **left sidebar**:

### 📁 Model Management
- Select from all available `.keras` checkpoints
- Model automatically loads when selected
- View layer count, parameters, and classes

### 📥 Dataset
- Download the FER2013 dataset with one click
- View per-split image counts

### 🏋️ Train Model
- Adjust **epochs** (5–100) and **batch size** (16–128)
- Training runs in a **background subprocess** — UI stays responsive
- Live training log streaming
- New model is auto-timestamped: `emotion_model_YYYYMMDD_HHMMSS.keras`

### 📷 Live Test *(Local Only)*
- Open webcam and test predictions in real time
- Disabled automatically on Streamlit Cloud

---

## 🏋️ Training

### From the Web UI
The easiest way to train:
1. Download dataset via sidebar
2. Set epochs & batch size
3. Click **Start Training**
4. Watch the log in the **Training Monitor** section

### From Command Line
```bash
# Default settings (50 epochs, batch 64)
python train_model.py

# Custom settings
python train_model.py --epochs 30 --batch-size 32 --model my_model.keras
```

**Callbacks:**
- `EarlyStopping` — stops if validation loss plateaus (patience=8)
- `ModelCheckpoint` — saves the best model
- `ReduceLROnPlateau` — reduces learning rate on plateau

### Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D (32) + BN + Conv2D (32) + BN + MaxPool + Dropout | Input block |
| Conv2D (64) + BN + Conv2D (64) + BN + MaxPool + Dropout | |
| Conv2D (128) + BN + Conv2D (128) + BN + MaxPool + Dropout | |
| Conv2D (256) + BN + Conv2D (256) + BN + MaxPool + Dropout | |
| Flatten + Dense(256) + BN + Dropout(0.5) | |
| Dense(128) + BN + Dropout(0.5) | |
| Dense(7) Softmax | Output |

---

## 🌐 Streamlit Web App

Start the server:
```bash
streamlit run app.py
```

### Tabs

- **📷 Live Camera** — Real-time webcam feed with face detection & emotion overlay *(local only)*
- **🖼️ Upload Image** — Upload any image for prediction with side-by-side comparison and per-face rankings
- **📊 Model & Dataset** — Dataset distribution charts, model architecture table, checkpoint browser, and training history curves

---

## ☁️ Deploy to Streamlit Community Cloud

### 1. Push to GitHub
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Select `app.py` as the main file
4. Click **Deploy**

### Notes
- The app detects cloud environments and **disables webcam** automatically
- Training works on Cloud but is CPU-only (slower)
- For webcam testing, run the app **locally**
- System dependencies (`libgl1` for OpenCV) are declared in `packages.txt`

---

## 📷 Standalone Real-Time Script

Run without a browser:
```bash
python realtime_prediction.py
```

Press **ESC** to exit.

Options:
```bash
python realtime_prediction.py --model emotion_model.keras --camera 0
```

---

## 🎯 Emotion Classes

| # | Emotion | Emoji | Color |
|---|---------|-------|-------|
| 0 | Angry | 😠 | `#ff5252` |
| 1 | Disgust | 🤢 | `#69f0ae` |
| 2 | Fear | 😨 | `#e040fb` |
| 3 | Happy | 😊 | `#ffd740` |
| 4 | Sad | 😢 | `#448aff` |
| 5 | Surprise | 😲 | `#ff6e40` |
| 6 | Neutral | 😐 | `#b0bec5` |

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `emotion_model.keras` | Path to trained model |
| `DATASET_DIR` | `dataset` | Dataset root directory |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `BATCH_SIZE` | `64` | Training batch size |
| `EPOCHS` | `50` | Training epochs |

---

## 📜 License

MIT

---

## 🙏 Acknowledgments

- **FER2013 Dataset** — Pierre-Luc Carrier & Aaron Courville
- **OpenCV Haar Cascades** — Intel / OpenCV team
- **TensorFlow / Keras** — Google
- **Streamlit** — Snowflake
