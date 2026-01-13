# AI-Based Public Safety Monitoring & Risk Detection System  
*(Prototype / Proof of Concept)*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Prototype-yellow.svg)

---

## 📌 Project Overview

Crowded public environments such as **metro stations**, **markets**, and **festivals** are vulnerable to sudden safety incidents like panic, stampedes, or abnormal crowd movement. Traditional CCTV monitoring relies heavily on human attention, which is slow, error-prone, and difficult to scale.

This project presents a **research prototype** that demonstrates how **computer vision and machine learning** can be used to analyze **crowd-level motion patterns** from video and detect **abnormal crowd behavior**, generating **event-based safety alerts**.

> ⚠️ This is a **prototype / proof of concept**, not a production deployment.

---

## 🎯 Problem Statement

Current surveillance systems face several challenges:

- Manual monitoring is slow
- Human operators may miss early warning signs
- Difficult to scale across many cameras
- Lack of automated, explainable alerts

---

## 💡 Proposed Solution

A **logic-driven AI prototype** that:
- Analyzes **crowd motion only** (no individuals)
- Detects **abnormal collective behavior**
- Predicts **future risk** using temporal modeling
- Generates **event-based alerts** with explanations

---

## ⚖️ Ethical AI Compliance

This system strictly follows privacy-first principles:

| Aspect | Status |
|------|-------|
| Face recognition | ❌ Not used |
| Individual tracking | ❌ Not used |
| Biometric data | ❌ Not stored |
| Analysis level | ✅ Crowd-level only |

---

## 📂 Dataset

**UMN Crowd Activity Dataset**
- Source: University of Minnesota
- Type: Pre-recorded surveillance-style videos
- Content:
  - Normal crowd walking
  - Abnormal panic / running behavior

**Video used:**
- `Crowd-Activity-All.avi`
- Converted to `crowd_15fps.mp4` (15 FPS)

---

## 🧠 System Workflow

The project is implemented as a **multi-stage offline pipeline**:

1. Motion feature extraction
2. Dataset labeling (weak supervision)
3. Temporal ML training
4. Event-based alert generation
5. Performance evaluation

---

## 🔍 Motion Feature Extraction

**Technique:** Dense Optical Flow (Farneback)

Why optical flow?
- Captures **collective motion**
- Does not identify people
- Computationally efficient on CPU

**Extracted features per frame:**
- `frame_id`
- `mean_motion`
- `variance_motion`

These represent:
- Overall crowd movement intensity
- Instability / chaos in motion

---

## 🏷️ Weakly Supervised Labeling

Manual annotation is avoided using **data-driven rules**:

Percentile thresholds on `mean_motion`:
- Below 60th percentile → Normal
- 60th–85th percentile → Medium risk
- Above 85th percentile → High risk (abnormal)

Generated labels include:
- `normal_activity`
- `abnormal_activity`
- `normal_alert`
- `medium_alert`
- `high_alert`

---

## ⏱️ Temporal Modeling & Leakage Prevention

To avoid data leakage:
- The model predicts **future abnormal behavior**
- `high_alert` is shifted by **one frame ahead**
- Ensures honest evaluation on time-series data

Target variable:
- `future_high_alert`

---

## 🤖 Machine Learning Model

- Algorithm: **Random Forest Classifier**
- Features:
  - Mean motion
  - Motion variance
- Output:
  - Binary prediction of future abnormal behavior

**Why supervised learning?**
- Limited dataset size
- Better interpretability
- Lower data requirements than deep learning

---

## 🧠 Decision Logic (Safety-First)

Public safety systems prioritize **recall over precision**.

Implemented strategies:
- Lower probability threshold (0.4)
- Temporal OR smoothing (3-frame window)

This reduces missed dangerous events while controlling noise.

---

## 🚨 Event-Based Alert Generation

Instead of frame-by-frame alerts, consecutive abnormal frames are grouped into **events**.

Each alert reports:
- Start & end timestamp
- Frame range
- Risk level
- Human-readable cause

**Example:**
```
ABNORMAL CROWD ACTIVITY DETECTED
Time Range : 00:03:49 → 00:03:50
Frame Range: 3446 → 3452
Risk Level : HIGH
Cause : Sustained abnormal crowd behavior
```

---

## 📊 Performance Evaluation

### Confusion Matrix

| | Predicted Normal | Predicted Abnormal |
|--|--|--|
| Actual Normal | 593 | 176 |
| Actual Abnormal | 19 | 373 |

### Key Metrics (Abnormal Class)

| Metric | Value |
|------|------|
| Accuracy | 83% |
| Precision | 0.68 |
| Recall | **0.95** |
| F1-score | 0.79 |
| False Alert Rate | 22.9% |

**Interpretation:**
- Very high recall ensures most dangerous events are detected
- Some false alerts are acceptable in safety contexts

---

## ▶️ How to Run

### Step 1: Extract motion features
```bash
python extract_motion.py
```

### Step 2: Convert features to labeled CSV
```bash
python numpy_to_csv.py
```

### Step 3: Run the ML pipeline
```bash
python final_pipeline.py
```

---

## 📁 Project Structure
```
crowd-dataset/
├── extract_motion.py
├── numpy_to_csv.py
├── final_pipeline.py
├── crowd_15fps.mp4
├── motion_features.npy
├── labeled_motion_features.csv
├── README.md
```

---

## 🚀 Future Enhancements (Post-Selection)

- Minimum-duration filtering to remove very short spikes
- Confidence scores for alerts
- Crowd density estimation
- Multi-camera fusion
- Live CCTV stream integration
- Operator dashboard visualization

---

## 👨💻 Author

**Mrityunjay Singh**  
AI / ML Engineer  
Computer Vision & Public Safety Systems

---

## 🏁 Conclusion

This prototype demonstrates how ethical AI, classical computer vision, and machine learning can be combined to build an explainable crowd safety monitoring system. The project focuses on correctness, transparency, and safety rather than over-automation, making it suitable for further research and development.