# 🧠 Computational Multimodal Cognitive Load Detection System

## Overview

The Multimodal Cognitive Load Detection System is a Python desktop application that estimates a student's cognitive load while answering quiz questions. It combines information from facial behavior, mouse movement, and user interaction to classify cognitive load as **Low**, **Medium**, or **High**.

> **Note:** This project is based on **rule-based heuristics and computer vision**, that is, a computational model. The ADHD and Dyslexia results are only behavioral indicators for educational purposes and are **not medical diagnoses**.

---

# Features

- 📹 Live webcam face and eye detection
- 👁️ Real-time blink tracking
- 🖱️ Mouse speed, movement, and idle tracking
- ⏱️ Response time measurement
- 📊 Multimodal cognitive load prediction
- 🎯 Automatic quiz scoring
- 📈 End-of-session analysis report
- ⚠️ ADHD and Dyslexia behavioral indicators
- 🌙 Dark and Light mode
- 💾 SQLite database for session history

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3 | Programming language |
| Tkinter | Desktop GUI |
| OpenCV | Webcam, face and eye detection |
| Haar Cascades | Face and eye detection model |
| Pillow (PIL) | Display webcam frames in Tkinter |
| NumPy | Mathematical calculations |
| SQLite3 | Store session history |
| Threading | Background webcam processing |

---

# Project Structure

```
cognitionload_detector/
│
├── data/
│   └── cognitive_load.db
│
├── src/
│   ├── main.py
│   ├── face_analyzer.py
│   ├── mouse_tracker.py
│   ├── interaction_logger.py
│   ├── predictor.py
│   ├── analyzer.py
│   ├── database.py
│   └── models.py
│
├── run.py
├── requirements.txt
└── README.md
```

---

# How It Works

The system collects information from three different sources.

### 1. Facial Analysis
Using OpenCV Haar Cascade classifiers, the application detects:
- Face presence
- Eye state
- Blink rate
- Head position

### 2. Mouse Analysis
The application records:
- Cursor speed
- Cursor jitter
- Idle time

### 3. User Interaction
The application measures:
- Response time
- Quiz progress
- Question difficulty
- Answer correctness

These features are combined using a **rule-based multimodal fusion system** to estimate the student's cognitive load.

---

# Cognitive Load Levels

The system predicts one of three levels:

- 🟢 Low
- 🟡 Medium
- 🔴 High

The prediction is based on facial behavior, mouse activity, response time, blink rate, and question difficulty.

---

# Analysis Report

At the end of the quiz, the system generates a report containing:

- Overall score
- Accuracy
- Average cognitive load
- Facial behavior summary
- Blink analysis
- Mouse movement analysis
- Performance by question difficulty
- Question-by-question results
- ADHD and Dyslexia indicators
- Personalized recommendations

---

# Installation

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python run.py
```

---

# Requirements

- Python 3.9+
- Webcam
- Windows/Linux/macOS

---

# Future Improvements

This project currently uses rule-based decision making. It can be extended into a true machine learning system by:

- Collecting labeled cognitive load data
- Using MediaPipe or OpenFace for facial feature extraction
- Training machine learning models using Scikit-learn or TensorFlow
- Evaluating performance using train/test datasets

---

# Ethical Note

This project is intended for educational and research purposes only. It should not be used to diagnose ADHD, Dyslexia, or any medical condition.

---

# One-Line Summary

**A Python desktop application built with Tkinter, OpenCV, and SQLite that estimates student cognitive load in real time using facial behavior, mouse movement, and interaction analysis through a rule-based multimodal fusion approach.**
