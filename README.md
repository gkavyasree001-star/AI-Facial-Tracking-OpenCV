# AI-Facial-Tracking-OpenCV
A lightweight computer vision pipeline built using Python and OpenCV that captures live webcam feeds, maps face regions using Haar Cascades, and applies geometric feature tracking to estimate expressions with zero latency.
# Real-Time Facial Feature & Expression Tracking

A high-performance computer vision pipeline built using native OpenCV to detect human faces and dynamically approximate facial states in real-time. 

This project is explicitly engineered using lightweight geometric Haar Cascade classifiers to ensure zero-latency frame processing and native stability across modern Python environments without heavy deep-learning dependency overhead.

---

## 🎯 Project Features
- **Native Edge Processing:** Runs entirely on CPU without requiring heavy CUDA/GPU acceleration.
- **Multi-Feature Cascade Matrix:** Simultaneously runs regional face, eye, and smile detection layers.
- **Dynamic State Estimation:** Maps facial box aspect ratios and landmark tracking frequencies to display live tracking states (HAPPY, SURPRISE, NEUTRAL, etc.).
- **Horizontal Mirroring:** Automated frame flipping for a natural user interface layout.

---

## 🛠️ Tech Stack
* **Language:** Python 3.11+ (Fully optimized for Python 3.14 compatibility)
* **Core Framework:** OpenCV (Open Source Computer Vision Library)
* **Execution Environment:** Isolated Virtual Env (`venv`)

---

## ⚙️ Quick Local Setup

### 1. Position into Directory
```bash
cd C:\AI_Interview_Projects\2_Facial_Expression
