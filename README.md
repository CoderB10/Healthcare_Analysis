# Healthcare Forecasting Comparative Study 

This repository compiles five detailed research case studies exploring diverse approaches to **healthcare forecasting**, from pandemic hospital capacity planning to AI‑based clinical predictions. Each study showcases a distinct modeling strategy, dataset, and operational insight.

## Included Papers

1. **On‑request Local Hospital Bed Forecasting for COVID‑19 (Germany)**
   *Local simulation‑based forecasting for pandemic response.*
2. **DT‑GPT | Large Language Model Digital Twin for Clinical Forecasting**
   *LLM‑based patient health trajectory forecasting from raw EHRs.*
3. **CDPred | Forecasting Cognitive Decline from Wearable Activity Data**
   *Machine learning on sensor data for dementia risk prediction.*
4. **Hybrid Data‑Mined Clinical Pathway Simulation for Acute Coronary Syndrome**
   *Data‑driven discovery of hospital pathways combined with discrete‑event simulation.*
5. **Validating Local Hospital AI Models Using Global Benchmark Data (MIMIC‑IV Study)**
   *Cross‑dataset benchmarking of hospital admission predictors for model generalizability.*

---

### Paper: DT-GPT | Large Language Model Digital Twin for Clinical Forecasting

**Citation:** Authors TBD, Year TBD, preprint/venue TBD


**Theme:**

This paper delves into how a fine‑tuned large language model, DT‑GPT, transforms longitudinal patient data into an interpretable, text‑based forecasting engine. It treats the entire electronic health record as natural language, enabling the model to predict lab trajectories and cognitive scores even with severe data sparsity. The system introduces an explainable and conversational interface, allowing clinicians to query the model’s reasoning and perform zero‑shot predictions on unseen variables, making it a step forward toward deployable patient digital twins.

**Execution (step-by-step):**

### 1. Data Cleaning (or Lack Thereof)

**What DT-GPT Did:**

* **No Imputation:** Trained directly on missing values; no guessing of data.
* **No Normalization:** Used raw value scales; model learned contextually.
* **Outlier Filtering:** Minimal two-step filter, remove 3 SD outliers, recalculate SD, then clip extremes.

**What Baselines Did:**

* All 14 baseline models (LightGBM, TFT, etc.) required z-score normalization and imputation.

---

### 2. Core Calculation: Encoding & Fine-Tuning

* **Encoding:** Entire patient EHR turned into text prompt:
  * Chronological list of visits, labs, diagnoses
  * Static demographics
  * Forecast dates and prediction instruction
* **Base Model:** BioMistral‑7B (biomedical LLM)
* **Fine-Tuning:** Supervised instruction-tuning on EHR text → output JSON forecast of lab/vital values.
* **Loss Function:** Cross-entropy loss applied only on output text.
* **Stabilization:** 30 trajectories generated per patient; averaged for final prediction.

---

### 3. Baseline Models (14 Total)

**Categories:**

* **Simple:** Copy Forward (last value persists)
* **Classical ML:** LightGBM, Linear Regression
* **Deep Learning:** RNN, LSTM, Transformer, Temporal Fusion Transformer (TFT)
* **Zero-Shot LLMs:** BioMistral‑7B (untrained), Qwen3‑32B — both poor
* **Channel-Independent LLMs:** Time‑LLM, LLMTime — weaker due to independent variable prediction

**Advantage:** DT‑GPT is *channel‑dependent*, learning cross‑variable biological relations.

---

### 4. Evaluation (Reliability Tests)

* **Data Split:** 80/10/10 by patient (ensures unseen patients at test time)
* **Primary Metric:** Scaled MAE (MAE ÷ SD) — normalizes across lab value scales
* **Secondary Metrics:**

  * **KS Statistic:** distribution alignment between forecast and real data
  * **AUC ROC:** classification accuracy for clinically relevant events (e.g., anemia detection)

**Robustness Tests:**

* **Missingness Test:** Added +20 % missingness → stable performance
* **Noise Test:** Injected misspellings → degradation only after 25+ errors/patient
* **Data Size Test:** Scaled down to 5 k patients → competitive accuracy retained

**Zero-Shot Forecasting:**

* Trained on 6 NSCLC labs, tested on 69 unseen ones
* Beat fully‑trained LightGBM on 13 of 69 (esp. biologically related metrics)

---

**Results Summary:**

* **Accuracy:** RMSE improved 1.3 – 3.4 pp vs. best baselines
* **Robustness:** Stable under extreme missingness (94 %) and noise
* **Explainability:** Natural‑language rationale ~80 % correct by physician review

**Operational Benefits:**

* Eliminates preprocessing pipeline
* Handles messy, sparse, unnormalized data
* Provides interpretable forecasts and zero‑shot inference via chat interface

**Robustness & Limits:**

* High compute (7 B parameters)
* Internal attention explainability limited
* Deployment on small infrastructure untested

**Repro Notes:**

* **Code/Repo:** BioMistral‑7B base; fine‑tuning scripts pending release
* **Data Access:** Restricted medical datasets (NSCLC, ICU, ADNI)

**Takeaways:**

* LLM‑based forecasting thrives on messy real‑world data.
* Channel‑dependent modeling learns inter‑variable biology.
* DT‑GPT advances digital‑twin realism via explainable, zero‑shot forecasts.

---

### Paper: CDPred | Forecasting Cognitive Decline from Wearable Activity Data

**Citation:** Authors TBD, Year TBD, Journal TBD
**Scenario Tags:** disease-incidence, cognitive-decline, mortality-risk, wearable-sensing, early-screening

**Theme:**

This research investigates the predictive potential of everyday human movement captured via wearable sensors. Instead of relying on medical imaging or invasive testing, it decodes the rhythm and regularity of 24‑hour motion signals to forecast cognitive deterioration years in advance. The work demonstrates that harmonic and entropy‑based features extracted from accelerometer data hold meaningful physiological insights, enabling a non‑invasive, scalable biomarker for early dementia screening.

**Execution (step-by-step):**

### 1. Data Cleaning (or Lack Thereof)

* **Filtering Rules:**

  * If baseline clinical data were missing → participant excluded.
  * If sensor data <1 valid day (≥10 daytime hours) → participant excluded.
* Result: smaller, cleaner subset with complete clinical + valid activity data.
* No imputation beyond these inclusion filters.

### 2. The Calculations (The "Secret Sauce")

* **Standardization:** Converted raw accelerometer output to minute-by-minute activity metrics (ENMO, VMC).
* **Harmonic Features:** Treated each 24‑hour cycle as a waveform, applied Fast Fourier Transform (FFT) and entropy measures to derive **98 harmonic features** quantifying:

  * **Rhythm:** stability of daily 24‑hour pattern
  * **Entropy:** randomness or predictability of movement
  * **Shape:** burstiness (kurtosis) and asymmetry (skewness)
* This signal‑processing layer captured movement *quality* rather than quantity.

### 3. Forecast Model (XGBoost Classifier - "CDPred")

* **Algorithm:** Gradient‑boosted decision trees (XGBoost)
* **Target:** Binary, cognitive decline (ΔMoCA < 0) vs. stable (ΔMoCA ≥ 0)
* **Inputs:** demographic + clinical + 98 harmonic features
* **Variants:**

  * **CDPred:** demographics + clinical
  * **CDPred‑4:** + simple activity volume
  * **CDPred‑4+:** + 98 harmonic features
* **Hyperparameters:** max_depth=6, learning_rate=0.05, 500 trees; tuned via 5‑fold CV

### 4. Evaluation (Reliability Tests)

* **Hold‑Out Sets:** 10 % of hip cohort, 15 % of wrist cohort reserved for testing.
* **Comparative Models:** 3‑way bake‑off,  baseline vs. simple v/s full model.
* **Metrics:**

  * **Accuracy:** proportion correctly classified
  * **AUC (ROC):** discrimination ability (0.5=random, 1.0=perfect)

**Results:**

| Cohort      | Model                 | Accuracy | AUC      |
| ----------- | --------------------- | -------- | -------- |
| Hip (1 y)   | Demographics only     | 75 %     | 0.74     |
| Hip (1 y)   | Full model (movement) | **84 %** | **0.86** |
| Wrist (5 y) | Demographics only     | 66 %     | 0.65     |
| Wrist (5 y) | Full model (movement) | **69 %** | **0.73** |

* **Best Model:** CDPred‑4+ Due to it's large, consistent improvement across sensors & timelines.
* **Reliability Proof:** Replicated success across distinct cohorts (hip vs. wrist), wear durations, and forecast horizons (1 y / 5 y) → robust biomarker signal.

**Operational Gains:**

* Non‑invasive early screening method
* Requires only short wearable recordings (72h - 7d)
* Enables scalable, at‑home risk assessment for cognitive decline

**Robustness & Limits:**

* Wrist data noisier; shorter wear reduces precision
* Small cohorts; demographic balance unverified
* Potential gains from adding genetic or neuroimaging data

**Repro Notes:**

* **Code/Repo:** not public
* **Data Access:** private clinical cohorts; IRB‑approved

**Takeaways:**

* Movement *rhythm*, not volume, best predicts cognitive decline.
* Harmonic features strongly outperform classic risk factors.
* Consistent results across devices & timescales confirm generalizable signal.

---

### Paper: Hybrid Data-Mined Clinical Pathway Simulation for Acute Coronary Syndrome

**Citation:** Authors TBD, Year TBD, Venue TBD
**Scenario Tags:** hospital-admissions, patient-flow, resource-allocation, length-of-stay, simulation-modeling

**Theme:**

This study integrates data mining and simulation science to make hospital process modeling more representative of clinical reality. By clustering thousands of patient journeys into data‑derived pathways and embedding them into a discrete‑event simulation, it captures the true variability of patient movement through departments. The hybrid model not only enhances predictive realism of length‑of‑stay and queue times but also serves as a strategic decision tool for capacity management.

**Execution (step-by-step):**

### 1. Data Cleaning (Handling the Mess)

* **Dataset:** 3,434 Electronic Health Records (EHRs) for Acute Coronary Syndrome (ACS) from Almazov Centre, Russia.
* **Sorting:** 89.3 % of patient logs were out of chronological order → sorted by timestamp.
* **Reconstruction:** 11 % had missing events → reconstructed logically (e.g., inferred department exits).
* **Outcome:** Complete, chronological event logs for all patients.

### 2. Discovery Calculations (Finding the Real Paths)

* **Encoding:** Translated each patient's sequence of department visits into a text string (e.g., Admission→ICU→Surgery→ICU→Cardiology → `AFIFE`).
* **Clustering Model:** K-means with **Levenshtein distance** to group similar pathways.
* **Output:** Identified **10 unique clusters**, each representing a distinct real-world clinical pathway.

  * Cluster 5 → textbook cases
  * Cluster 2 → inter-hospital transfers
  * Clusters 3 & 9 → high-risk, complex trajectories

### 3. Forecast Model (Simulating Reality)

* **Simulation Type:** **Discrete-Event Simulation (DES)** using **SimPy (Python)**.
* **Hybrid Twist:** Simulation incorporates 10 distinct patient types (one per cluster).
* **Execution Flow:**

  1. Patient generator spawns new entities.
  2. Each patient randomly assigned a cluster based on empirical probability.
  3. Follows its pathway’s department-transition rules and length-of-stay distributions.
  4. Requests hospital resources (beds, surgery rooms); queues if unavailable.
* **Forecast Output:** Aggregate hospital metrics—bed occupancy, queue lengths, average wait time, length-of-stay (LoS) distribution.

### 4. Evaluation (Proving Reliability)

* **Baseline Model:** Simple DES using overall hospital averages (no clusters).
* **Comparison Metrics:**

  * **Kolmogorov–Smirnov Test:** hybrid model 51 % closer to real LoS distribution vs. baseline.
  * **QQ-Plot:** baseline failed for long-stay (>20 days) cases; hybrid captured long-tail behavior.
* **Scenario Analysis (Practicality):** Reduced surgery rooms from 4 → 3.

  * **Result:** wait time ↑ from <1 h → ~3 h; queued patient share spiked sharply.

**Results Summary:**

* **Realism:** +51 % fit improvement to real data.
* **Predictive Power:** Accurately modeled rare, high-complexity patient cases.
* **Utility:** Produced clear, interpretable what-if insights for hospital managers.

**Robustness & Limits:**

* One-center dataset (ACS-specific); generalization to other conditions pending.
* Computationally intensive (K-means + DES).
* Transferability to multi-disease or multi-hospital contexts untested.

**Repro Notes:**

* **Code/Repo:** SimPy-based Python prototype; not publicly released.
* **Data Access:** Restricted hospital EHRs (Almazov Centre).

**Takeaways:**

* Hybrid modeling bridges descriptive data mining and operational forecasting.
* Cluster-based DES captures true patient heterogeneity.
* Enables actionable scenario planning (capacity, queue management).

---

### Paper: Validating Local Hospital AI Models Using Global Benchmark Data (MIMIC-IV Study)

**Citation:** Authors TBD, Year TBD, Journal TBD
**Scenario Tags:** hospital-admissions, resource-allocation, machine-learning-validation, generalizability, model-benchmarking

**Theme:**

The paper examines AI model generalizability across scales by replicating a small‑hospital admission‑prediction study on a vastly larger global benchmark, MIMIC‑IV. It provides a controlled comparison between localized training and large‑scale validation, illustrating how algorithmic performance shifts with data volume and diversity. The findings underscore that robust, public datasets can validate or refine smaller hospital models while exposing the pitfalls of overfitting when accuracy appears “too perfect.”

**Execution (step-by-step):**

### 1. Data Cleaning and Preparation (The "How")

* **Dataset:** 322,189 patient records from **MIMIC-IV** (Beth Israel Deaconess Hospital, Boston)
* **Variable Extraction:** matched variables from Greek local dataset (age, gender, 13 lab markers: CRP, WBC, Creatinine, etc.)
* **Outlier Removal:** applied **Tukey’s method** (±1.5×IQR) to drop extreme values (e.g., impossible vitals).
* **Missing Data Handling:** filled gaps using **median imputation** (feature-wise median replacement).
* **Categorical Encoding:** applied **One-Hot Encoding** to convert text (e.g., gender) into numeric binary features.
* **Outcome:** standardized, clean dataset identical in structure to Greek hospital data (13,991 cases).

### 2. Forecast Models Used (The Contestants)

Five classic ML models trained and evaluated on the MIMIC-IV subset:

1. **Linear Discriminant Analysis (LDA):** linear separation of admission vs. discharge.
2. **K-Nearest Neighbors (KNN):** classifies based on closest k patient profiles.
3. **Recursive Partitioning (RPART):** decision-tree using if–then logic.
4. **Support Vector Machine (svmRadial):** non-linear separation via radial kernel.
5. **Random Forest (RF):** ensemble of decision trees with majority voting — eventual winner.

### 3. Model Training and Calculation

* **Method:** **10-fold cross-validation**

  * Split data into 10 folds (~32,200 cases/fold)
  * Each fold alternates as test data once; trained on remaining nine folds
  * Aggregate 10 results → mean performance per model
* **Purpose:** prevent overfitting and ensure reliable generalization.

### 4. Evaluation Metrics

* **Primary Metric:** Area Under ROC Curve (AUC-ROC)
* **Secondary Metrics:**

  * **Sensitivity:** correctly identified admitted patients
  * **Specificity:** correctly identified discharged patients

**Results:**

| Algorithm     | AUC (Greek) | AUC (MIMIC-IV) | Notes                           |
| ------------- | ----------- | -------------- | ------------------------------- |
| Random Forest | **0.8054**  | **0.9999**     | Champion; near-perfect accuracy |
| LDA           | 0.7834      | 0.9387         | Strong performer                |
| RPART         | 0.6989      | 0.9092         | Improved on large data          |
| svmRadial     | 0.7961      | 0.7640         | Slight degradation              |
| KNN           | 0.7307      | 0.7112         | Weakest model                   |

* **Random Forest Performance:**

  * **AUC:** 0.9999
  * **Sensitivity:** 99.97 %
  * **Specificity:** 99.99 %
  * Indicates robust pattern recognition at massive data scale.

### 5. Interpretation & Reliability

* **Key Finding:** Random Forest consistently top across datasets.
* **Validation:** MIMIC-IV benchmarking confirmed Greek model’s correctness and robustness.
* **Overfitting Caveat:** 0.9999 AUC possibly inflated — memorization risk acknowledged by authors.

**Results Summary:**

* Random Forest validated as the optimal algorithm for hospital admission prediction.
* Larger dataset amplified accuracy dramatically (data-scale effect).
* Confirms feasibility of using open datasets for cross-site AI validation.

**Robustness & Limits:**

* Potential overfitting at near-perfect metrics.
* Tested only on tabular data (no imaging/text).
* External generalization beyond MIMIC-IV not verified.

**Repro Notes:**

* **Code/Repo:** not specified; reproducible via MIMIC-IV public access + Python ML stack.
* **Data Access:** MIMIC-IV (public academic license); Greek hospital dataset (private).

**Takeaways:**

* Large-scale validation is essential for confirming small-hospital AI reliability.
* MIMIC-IV can serve as a benchmarking standard for clinical ML validation.
* Random Forest remains robust, interpretable, and reliable across dataset scales.

---
