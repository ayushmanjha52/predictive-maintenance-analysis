Model Card: Combi Mill Field Device Cause Classifier
Project: Tata Steel Combi Mill Predictive Maintenance
Developer: Ayushman Jha
Last updated: July 2026 — regenerate this card every time train.py/retrain_model.py runs
Model Version: v2.0 (Random Forest, retuned)

1. Model Purpose
Predicts the most likely field device causing a process delay, from the
free-text delay description logged by maintenance staff. Covers HMD,
Photocell, Encoder, LVDT, Proximity, Pressure Switch, Flow Switch, Laser,
and Proximity Switch.

2. Model Details
The deployed model is a Random Forest Classifier, selected via comparison
against Logistic Regression and HistGradientBoosting — see
models/training_manifest.json for the full comparison. Its hyperparameters
are n_estimators=264, max_depth=None, max_features="sqrt", min_samples_split=8, min_samples_leaf=1, class_weight="balanced_subsample",
found via hyperparameter_tuning.py using a 30-iteration RandomizedSearchCV.
Training data consists of 345 labeled delay events spanning Nov'25 through
May'26, across 9 device classes. The model uses 168 numeric features: TF-IDF
text features combined with domain-specific keyword and pattern flags.
Model selection was done by comparing multiple model families and choosing
the winner based on honest time-based holdout accuracy — training on all
months except the most recent and testing only on that unseen month — not
by cross-validation score alone.

3. Accuracy — read this section before trusting any single number
Two different accuracy numbers exist for this model, and they disagree by
about 6 points. Use the second one.
The 5-fold CV weighted F1 score is 0.7634. This is optimistic, because the
folds mix all months together, so the model sees similar phrasing from
every month during training. The time-based holdout accuracy — training on
Nov'25 through Apr'26 and testing only on the unseen May'26 month (n=37) —
is 0.7027. This is the honest number, since it reflects what actually
happens every month going forward.
Report 70.3% as the headline accuracy, not 76.3%. This project has
repeatedly found that CV score alone is not a reliable indicator of
real-world performance — for earlier model versions the gap was as large as
25 points (76.5% CV vs. 51.4% holdout). This is why model selection is done
on holdout accuracy, not CV score, throughout this project.

4. Per-class performance
Regenerate this section from reports/classification_report.csv after
every retrain — do not carry over numbers from a previous model version.
Run this and paste the current per-class F1 scores here:

Code
python -c "import pandas as pd;
print(pd.read_csv('reports/classification_report.csv', index_col=0).round(2))"

One known, persistent weak spot has held across every model version tried
so far — Logistic Regression, CatBoost, HistGradientBoosting, and multiple
Random Forest configurations: Encoder consistently scores lowest, with F1
ranging from 0.38 to 0.52 depending on model version. Treat Encoder
predictions with extra caution regardless of which model is currently
deployed. This appears to be a genuine data/vocabulary limitation — Encoder
failure descriptions overlap linguistically with general billet/pass
terminology — rather than a tuning problem a better model will fix on its
own.

5. Known Limitations
Encoder predictions are less reliable than other devices, as noted above,
and this has held across every model version tried. The model is text-only
and has no access to live sensor/PLC signals from iba, so it cannot catch
degradation before it's described in a delay log. Training data is limited
to 345 labeled events across 9 classes, a small sample size, and some rare
failure modes — such as a class with only one example — were dropped
entirely during training. Feature engineering relies on keyword and pattern
matching and may miss genuinely new failure phrasing not seen in training
data. The model was trained on Nov'25 through May'26 data; re-run
hyperparameter_tuning.py and retrain_model.py periodically as new months
accumulate, and re-verify the honest holdout accuracy each time rather than
assuming it holds indefinitely. The model has also not yet been validated
in shadow mode against real plant outcomes — see Section 6.

6. When NOT to Trust the Model
Predictions with top confidence below 35% (config.CONFIDENCE_THRESHOLD)
are flagged low_confidence: true by the API — treat these as "manual
diagnosis needed," not a guess. Do not rely on Encoder predictions without
manual verification, regardless of the confidence shown. Do not use this
model for fully automated maintenance actions without human review — it
has not been validated in shadow mode yet (running in parallel with real
operations and comparing predictions against confirmed outcomes), which is
the recommended next validation step before any production trust. Finally,
a model claiming near-100% accuracy would actually indicate overfitting
given the CV-vs-holdout gap established above — never treat a very high
number as a sign of a better model without checking the holdout figure
first.

7. How to Use
Via the API, send a POST request to /predict with a JSON body containing
delay_text, for example {"delay_text": "HMD cleaning required in BDM"}.
Check /model_info for live, current model metadata — it reads
training_manifest.json directly, so it can never go as stale as this
document can if you forget to update it.
After any retrain, update Sections 2 through 4 from
models/training_manifest.json and reports/classification_report.csv.
Do not hand-edit accuracy numbers without re-checking the source files —
this exact card went stale once already, when v1.0 carried 334 samples, 164
features, and referenced a Random Forest that was no longer even deployed.

