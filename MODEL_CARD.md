# Model Card: Combi Mill Field Device Cause Classifier

**Project:** Tata Steel Combi Mill Predictive Maintenance  
**Developer:** Ayushman Jha  
**Date:** July 2026  
**Model Version:** v1.0 (Random Forest)

---

## 1. Model Purpose

To predict the most likely **field device** causing process delays in the Combi Mill using historical delay text data.  
This helps maintenance teams quickly identify probable root causes (HMD, Photocell, Encoder, LVDT, etc.).

---

## 2. Model Details

| Attribute              | Details                              |
|------------------------|--------------------------------------|
| **Model Type**         | Random Forest Classifier            |
| **Algorithm**          | `RandomForestClassifier` (scikit-learn) |
| **Training Data**      | 334 labeled delay events            |
| **Features**           | 164 numeric features (TF-IDF + domain-specific) |
| **Cross Validation**   | 5-Fold Stratified K-Fold            |
| **Final Accuracy**     | **73%** (Weighted F1 = 0.72)        |

---

## 3. Performance Summary

### Strong Performing Classes

| Device            | F1-Score | Remarks                  |
|-------------------|----------|--------------------------|
| **HMD**           | 0.78     | Very reliable            |
| **PHOTOCELL**     | 0.81     | Very reliable            |
| **LVDT**          | 0.72     | Good                     |
| **PRESSURE_SWITCH** | 0.88   | Excellent                |

### Weak Performing Classes

| Device     | F1-Score | Remarks                              |
|------------|----------|--------------------------------------|
| **ENCODER**| **0.38** | **Weak** – Use with caution          |
| **PROXIMITY** | 0.59   | Moderate                             |

> **Important:** The model currently performs poorly on **ENCODER**. Predictions for Encoder-related delays should be treated with low confidence.

---

## 4. Known Limitations

- Performance on **ENCODER** class is low (F1 = 0.38).
- The model only uses **textual descriptions** of delays. It does **not** use real-time sensor data from iba.
- Some rare failure modes were removed during training due to very low sample count.
- Feature engineering is based on keyword patterns and may miss new/unseen failure descriptions.
- Model was trained on data from Nov 2025 – May 2026. Performance may degrade over time without retraining.

---

## 5. When NOT to Trust the Model

- Do **not** fully trust predictions when the top confidence is below **40%** (low confidence flag is returned by the API).
- Avoid relying heavily on **ENCODER** predictions without manual verification.
- Do not use the model for fully automated maintenance decisions without human review.

---

## 6. How to Use

### Via API (Recommended)
```bash
POST /predict
{
  "delay_text": "HMD cleaning required in BDM"
}

