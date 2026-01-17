# Senta - Emotion AI Web App

This is a Flask-based web application for real-time facial emotion detection and micro-feature analysis.

## Features
- Webcam video capture (Client-side)
- Emotion Detection using Advanced CNN-LSTM
- Frame-by-frame Surface Vector analysis (Magnitude, Variance, Angle) for 6 facial regions.

## Setup
1. Ensure you have the required dependencies:
   ```bash
   pip install flask torch numpy opencv-python mediapipe
   ```

2. Run the application:
   ```bash
   python web_app/app.py
   ```

3. Open your browser and go to:
   `http://localhost:5000`

## Structure
- `app.py`: Backend server (Flask) + Model Inference
- `templates/index.html`: Frontend UI
- `static/js/main.js`: Client-side logic (Camera & Charts)
- `static/css/style.css`: Styling
