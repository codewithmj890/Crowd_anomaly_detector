# AI-Based Public Safety Monitoring & Risk Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

> **An ethical, explainable AI system that monitors crowd behavior in real-time to detect abnormal activities and prevent safety incidents in public spaces.**

---

## Problem Statement

Crowded public places such as **metro stations**, **markets**, and **festivals** are vulnerable to sudden safety incidents like panic, stampedes, or abnormal crowd movement. Current surveillance systems face critical limitations:

- **Manual CCTV monitoring is slow**
- **High dependency on human attention**  
- **Prone to missed incidents**
- **Not scalable across multiple camera feeds**

### Our Solution

An **automated AI-driven system** that analyzes crowd-level motion patterns in video to detect abnormal behavior and generate interpretable safety alerts in **near real-time**.

---

## Ethical AI Compliance

This system is designed with **privacy-first** principles:

| Feature | Status | Description |
|---------|--------|-------------|
| Individual Tracking | **Disabled** | No person identification |
| Face Recognition | **Disabled** | No facial data processing |
| Biometric Storage | **Disabled** | No personal data stored |
| Analysis Level | **Crowd-only** | Aggregate behavior analysis |
| Compliance | **Fully Compliant** | Ethical AI guidelines followed |

---

## Dataset & Methodology

### Dataset Information
- **Source**: University of Minnesota (UMN Crowd Activity Dataset)
- **Type**: Pre-recorded surveillance-style videos
- **Content**: Normal crowd walking + Abnormal panic/running behavior
- **Processing**: MP4 format, 15 FPS normalization

### Feature Extraction: Dense Optical Flow
We use the **Farneback method** for motion analysis:

```python
# Key advantages of our approach
• Captures collective crowd motion
• Does not identify individuals  
• Computationally efficient for CPU systems
```

**Extracted Features per Frame:**
- `mean_motion_magnitude` → Overall crowd movement intensity
- `variance_of_motion_magnitude` → Instability/chaos in crowd motion

---

## Machine Learning Pipeline

### 1. **Weakly Supervised Learning**
- **Method**: Percentile-based thresholding
- **Reason**: Manual annotation is expensive and time-consuming
- **Labels Generated**:
- **Normal Alert**: Low motion, stable crowd
  - **Medium Alert**: Moderate motion, potential risk  
  - **High Alert**: High motion, abnormal behavior

### 2. **Temporal Modeling & Leakage Prevention**
- **Issue Addressed**: Label leakage in time-series data
- **Solution**: Future-frame prediction (shift labels by 1 frame)
- **Benefit**: Ensures honest evaluation and real-world applicability

### 3. **Model Architecture**
- **Algorithm**: Random Forest Classifier
- **Input Features**: `[mean_motion, variance_motion]`
- **Target**: `future_high_alert`
- **Rationale**: Better interpretability + Lower data requirements

---

## Decision Logic Optimization

### Safety-First Design Philosophy
**Recall > Precision** → Better to have false alarms than miss critical events

| Technique | Implementation | Impact |
|-----------|----------------|---------|
| **Probability Threshold Tuning** | Lowered from 0.5 → 0.4 | Increased Sensitivity |
| **Temporal OR Smoothing** | 3-frame sliding window | Reduced Noise, Improved Event capture |

---

## Intelligent Alert System

### Event-Based Alerts (Not Frame-Based)
**Prevents alert flooding** and improves interpretability

```
ABNORMAL CROWD ACTIVITY DETECTED
Time Range : 00:03:49 → 00:03:50
Frame Range: 3446 → 3452  
Risk Level : HIGH
Cause      : Sustained abnormal crowd behavior
```

---

## Performance Metrics

### Confusion Matrix Results
```
                 Predicted
                Normal  Abnormal
Actual Normal    593     176
    Abnormal      19     373
```

### Key Performance Indicators

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 83.2% | Overall system correctness |
| **Precision (Abnormal)** | 67.9% | Alert reliability |
| **Recall (Abnormal)** | **95.2%** | Almost all dangerous events detected |
| **F1-Score (Abnormal)** | 79.1% | Balanced performance |
| **False Alert Rate** | 22.9% | Acceptable for safety systems |

### Why These Metrics Matter
- **High Recall (95.2%)**: Critical for safety - we catch almost all dangerous events
- **Low False Negatives (19)**: Very few critical events are missed
- **Moderate Precision**: Some false alerts are acceptable in safety-critical systems

---

## Quick Start

### Prerequisites
```bash
pip install opencv-python numpy pandas scikit-learn tqdm
```

### Run the Pipeline
```bash
python final_pipeline.py
```

### Expected Output
```
Extracting optical flow features...
Generating rule-based labels...

ALERTS
------------------------
ABNORMAL CROWD ACTIVITY DETECTED
Time Range : 00:03:49 → 00:03:50
Frame Range: 3446 → 3452
Risk Level : HIGH
Cause      : Sustained abnormal crowd behavior

PERFORMANCE METRICS SUMMARY
===============================================
Recall (Abnormal)    0.95    Ability to catch danger
False Alert Rate     22.90%  Normal events flagged
===============================================

Pipeline execution completed successfully.
```

---

## System Architecture

```mermaid
graph LR
    A[Video Input] --> B[Optical Flow]
    B --> C[Motion Features]
    C --> D[Rule-based Labels]
    D --> E[Random Forest]
    E --> F[Temporal Smoothing]
    F --> G[Alert Generation]
    G --> H[Safety Dashboard]
```

---

## Future Enhancements

### Short-term (Next 3 months)
- [ ] Minimum-duration filtering for noise reduction
- [ ] Confidence score assignment to alerts
- [ ] Improved alert severity scaling

### Medium-term (6-12 months)  
- [ ] Crowd density estimation
- [ ] Multi-camera fusion capabilities
- [ ] Adaptive thresholding based on environment

### Long-term (1+ years)
- [ ] Real-time deployment on live CCTV feeds
- [ ] Operator dashboard with visualization
- [ ] Integration with emergency response systems

---

## Project Structure

```
crowd-dataset/
├── final_pipeline.py          # Main production pipeline
├── crowd_15fps.mp4            # Input video data
├── labeled_motion_features.csv # Generated features
├── README.md                  # This file
└── requirements.txt           # Dependencies
```

---

## Key Achievements

- **End-to-end pipeline completed**  
- **Leakage-free evaluation validated**  
- **CPU-based deployment ready**  
- **Ethical AI compliance maintained**  
- **Safety-first design philosophy**  
- **Real-world applicability demonstrated**

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Mrityunjay Singh**  
*AI/ML Engineer specializing in Computer Vision & Public Safety Systems*

---

## 🙏 Acknowledgments

- University of Minnesota for the UMN Crowd Activity Dataset
- OpenCV community for optical flow implementations
- Scikit-learn team for machine learning tools

---

<div align="center">

**Built for Public Safety**

*Making crowded spaces safer through ethical AI*

</div>
=======

>>>>>>> 5c7e829801b19904ecbe43b9023f88cf0252ab95
