# Bank Marketing Machine Learning Preprocessing Pipeline

## Project Overview

This project implements a complete, end-to-end data exploration and preprocessing pipeline on the **Bank Marketing** dataset (`bank.csv`) to prepare clean, balanced, leakage-free training and testing datasets for machine learning classification models.

The target variable (`y`) indicates whether a bank client subscribed to a term deposit (`yes` / `no`). Because the raw dataset contains semicolon-delimited records, `"unknown"` placeholder values across several categorical features (such as `poutcome`, `contact`, `education`, and `job`), skewed numerical distributions, and class imbalance (~88.5% `no` vs. ~11.5% `yes`), this pipeline systematically cleans, engineers, encodes, scales, balances, and validates the data before modeling.

---

## Dataset Source

* **Dataset Name:** Bank Marketing (`bank.csv`)
* **Source:** UCI Machine Learning Repository
* **Dataset Link:** [https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* **Initial Dimensions:** 4,521 rows × 17 columns (`age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`, `y`)

---

## Steps in the Preprocessing Pipeline

### Stage 1: Dataset Loading, EDA & Data Cleaning - Saleem (Tasks 1–5)
1. **Import Libraries:** Load core data science and visualization libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`).
2. **Dataset Loading:** Read `bank.csv` using `;` as the delimiter and parse `"unknown"` variants as `NaN`.
3. **Initial Data Inspection:** Inspect head/tail rows, dataset shape (`4,521 × 17`), data types, summary statistics, missing value percentages, and duplicate rows (`0` duplicates found).
4. **Exploratory Data Analysis (EDA):** Analyze feature statistics (`age`, `balance`), category distributions (`job`, `marital`), and target subscription rate (`y`), supported by histograms and count plots.
5. **Data Cleaning:**
   * Identify missing values (`poutcome`: 81.95%, `contact`: 29.29%, `education`: 4.14%, `job`: 0.84%).
   * Drop the `poutcome` column due to excessive missingness.
   * Remove rows with missing `job` values (retaining 4,483 records) and verify data consistency and types.

### Stage 2: Feature Engineering, Train-Test Split & Statistical Analysis - Sivaranj(Tasks 6–10)
6. **Feature Engineering:**
   * Create `age_group` by binning `age` into six categories (`Young`, `Adult`, `Middle Age`, `Senior`, `Old`, `Very Old`).
   * Create `previously_contacted` (`No` if `pdays == -1`, otherwise `Yes`).
7. **Numerical Distribution Analysis:** Plot histograms with KDE curves and evaluate skewness across numerical features (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`).
8. **Categorical Analysis:** Compute value counts and generate count plots for categorical variables.
9. **Feature & Target Separation:** Separate predictor features (`X`) from the target variable (`y`).
10. **Data Leakage Check:** Identify and remove unnecessary or leakage-prone columns prior to model preparation.

### Stage 3: Train-Test Split & Data Preprocessing - Kevin Biju Kulangara (Tasks 11–15)
11. **Train-Test Splitting:** Split `X` and `y` into separate training and testing sets before fitting transformers to prevent data leakage.
12. **Numerical Data Preprocessing:** Separate numerical columns, impute missing numerical values, detect/treat outliers, and apply feature scaling (`StandardScaler`).
13. **Categorical Data Preprocessing:** Separate categorical columns, impute missing categorical values (`SimpleImputer`), and encode features using One-Hot Encoding, Ordinal Encoding, and Target Encoding where appropriate.
14. **DataFrame Concatenation:** Recombine the preprocessed numerical and encoded categorical DataFrames for both train and test splits.
15. **Label Encoding:** Encode the target variable (`y`) and applicable label columns numerically (`LabelEncoder`).

### Stage 4: Class Balancing, Feature Selection & Final Verification - Sharon (Tasks 16–20)
16. **Class Imbalance Handling:** Inspect class distribution and apply **SMOTE** strictly on the encoded training dataset to balance target classes without leaking into the test set.
17. **Correlation Matrix & Heatmap:** Compute and plot the feature correlation matrix heatmap to identify highly correlated features.
18. **Feature Selection:** Apply **Variance Threshold** to filter out low-variance or constant features.
19. **Final Verification:** Verify zero missing values, zero unexpected duplicates, proper numeric data types, consistent train/test shapes, balanced training classes, strict train-test separation, and no data leakage.
20. **Final ML-Ready Dataset:** Produce the finalized, preprocessed training and testing datasets ready for machine learning model training.

---

## Repository Structure

```text
<REPO-NAME>/                  
├── Final_bank_project.ipynb   # Combined end-to-end preprocessing notebook
└── README.md                     # Project documentation
