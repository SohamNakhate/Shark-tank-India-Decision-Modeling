# AI-Powered Shark Tank Decision Engine — Implementation Plan

## Background

Build an end-to-end ML system that analyzes the Shark Tank India dataset (750+ pitches, 80 columns, Seasons 1–5) to:

1. **Deal Prediction** — Binary classification: will a startup get a deal?
2. **Valuation Prediction** — Regression: estimate deal valuation for funded startups
3. **Shark Recommendation** — Multi-label classification: which sharks are likely to invest?

Six model families will be trained per task (where applicable), all tuned with **Optuna**:
Random Forest, Decision Tree,Naïve Bayes, Logistic Regression, Neural Network.

---

## User Review Required

> [!IMPORTANT]
> **Dataset download**: Kaggle requires authentication. You'll need to either:
> 1. Download the CSV manually from [Kaggle](https://www.kaggle.com/datasets/thirumani/shark-tank-india) and place it in `c:\projects\ML_Project\data\raw\`, **OR**
> 2. Provide your `kaggle.json` API key so I can download it programmatically.
>
> **Please confirm which approach you prefer before I begin execution.**

> [!NOTE]
> The Streamlit app will be the final deliverable. All intermediate model artifacts (`.pkl` files, Optuna study DBs) will be saved so the app loads pre-trained models at runtime.

---

## Proposed Architecture

```
ML_Project/
├── data/
│   ├── raw/                    # Original CSV(s)
│   └── processed/              # Cleaned & engineered features
├── notebooks/
│   ├── 01_eda_overview.ipynb           # Dataset overview, stats, distributions
│   ├── 02_eda_deal_analysis.ipynb      # Deal-specific deep dive
│   ├── 03_eda_shark_analysis.ipynb     # Shark investment patterns
│   └── 04_eda_feature_engineering.ipynb # Feature engineering exploration
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Data loading & validation
│   │   └── preprocessor.py     # Cleaning, encoding, feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py       # Abstract base class for all models
│   │   ├── deal_classifier.py  # Task 1: Deal prediction models
│   │   ├── valuation_regressor.py  # Task 2: Valuation estimation models
│   │   ├── shark_recommender.py    # Task 3: Multi-label shark recommendation
│   ├── tuning/
│   │   ├── __init__.py
│   │   └── optuna_tuner.py     # Unified Optuna tuning engine
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Custom metrics & evaluation helpers
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py    # Reusable plotting functions
│       └── helpers.py          # General utility functions
├── models/                     # Saved model artifacts (.pkl, .pt)
├── optuna_studies/             # Optuna study databases
├── app/
│   └── streamlit_app.py        # Streamlit web application
├── config.py                   # Project-wide configuration
├── requirements.txt
├── train.py                    # Main training orchestrator script
└── README.md
```

---

## Proposed Changes

### Component 1: Project Setup & Configuration

#### [NEW] [requirements.txt](file:///c:/projects/ML_Project/requirements.txt)
Key dependencies:
- `pandas`, `numpy`, `scikit-learn` — data & ML
- `optuna` — hyperparameter tuning
- `torch` (PyTorch) — neural networks
- `streamlit` — web app
- `matplotlib`, `seaborn`, `plotly` — visualization
- `joblib` — model serialization
- `jupyter`, `ipykernel` — notebooks
- `shap` — model interpretability (bonus)

#### [NEW] [config.py](file:///c:/projects/ML_Project/config.py)
Central configuration:
- File paths (raw data, processed data, model artifacts, Optuna DBs)
- Feature lists (numerical, categorical, target columns per task)
- Shark names list: `["Namita", "Vineeta", "Anupam", "Aman", "Peyush", "Ritesh", "Amit"]`
- Model hyperparameter search spaces
- Random seed, CV folds, Optuna trial counts

---

### Component 2: Data Pipeline

#### [NEW] [loader.py](file:///c:/projects/ML_Project/src/data/loader.py)
- `load_raw_data()` — reads CSV, validates schema, reports basic stats
- `load_processed_data(task)` — loads pre-processed data for a specific task
- Handles encoding issues, missing file errors gracefully

#### [NEW] [preprocessor.py](file:///c:/projects/ML_Project/src/data/preprocessor.py)
- **Cleaning**: Handle missing values (median for numeric, mode for categorical, flag columns for missingness)
- **Feature Engineering**:
  - `equity_to_valuation_ratio` = `Original Offered Equity / Valuation Requested`
  - `revenue_to_ask_ratio` = `Yearly Revenue / Original Ask Amount`
  - `profit_margin` = derived from Gross/Net Margin
  - `company_age` = current year - `Started in`
  - `presenter_diversity_score` = combination of male/female/transgender counts
  - `total_sharks_present` = sum of all `*_Present` columns
  - One-hot encode: `Industry`, `Pitchers State`, `Pitchers Average Age`
  - Label encode ordinal features
- **Task-specific preparation**:
  - **Task 1 (Deal)**: Target = `Received Offer` (binary). Use all features except deal-outcome columns.
  - **Task 2 (Valuation)**: Target = `Deal Valuation`. Filter to only rows where deal was made. Exclude deal-specific columns that leak target.
  - **Task 3 (Shark Recommendation)**: Target = binary vector `[Namita_invested, Vineeta_invested, ...]` derived from `*_Investment Amount > 0`. Filter to deal rows only.
- **Scaling**: StandardScaler for numerical features, fit on train set only
- **Train/test split**: Stratified 80/20 for classification; random 80/20 for regression

---

### Component 3: EDA Notebooks

#### [NEW] [01_eda_overview.ipynb](file:///c:/projects/ML_Project/notebooks/01_eda_overview.ipynb)
- Dataset shape, dtypes, missing value heatmap
- Descriptive statistics (mean, median, std for all numeric columns)
- Distribution plots for key financial features (Yearly Revenue, Monthly Sales, Ask Amount)
- Correlation matrix heatmap
- Season-wise pitch counts and deal rates

#### [NEW] [02_eda_deal_analysis.ipynb](file:///c:/projects/ML_Project/notebooks/02_eda_deal_analysis.ipynb)
- Deal vs. No-Deal comparison across features
- Industry-wise deal success rates
- Ask amount vs. deal probability
- Equity offered vs. deal probability
- Feature importance preview using basic Random Forest

#### [NEW] [03_eda_shark_analysis.ipynb](file:///c:/projects/ML_Project/notebooks/03_eda_shark_analysis.ipynb)
- Individual shark investment patterns (amount, frequency, preferred industries)
- Shark co-investment network (which sharks invest together most often)
- Shark deal amount distributions
- Shark presence vs. deal probability

#### [NEW] [04_eda_feature_engineering.ipynb](file:///c:/projects/ML_Project/notebooks/04_eda_feature_engineering.ipynb)
- Engineer and test new features
- Feature correlation with targets
- Feature selection analysis (mutual information, chi-squared)
- Demonstrate preprocessing pipeline

---

### Component 4: Model Development

#### [NEW] [base_model.py](file:///c:/projects/ML_Project/src/models/base_model.py)
Abstract base class providing:
- `train(X, y)`, `predict(X)`, `evaluate(X, y)` interface
- `get_optuna_search_space(trial)` — returns hyperparameters for a given Optuna trial
- `save(path)`, `load(path)` — serialization
- Common logging and metrics collection

#### [NEW] [deal_classifier.py](file:///c:/projects/ML_Project/src/models/deal_classifier.py)
**Task 1: Deal Prediction (Binary Classification)**

| Model | Key Hyperparameters (Optuna) |
|-------|------------------------------|
| Random Forest | n_estimators (50–500), max_depth (3–20), min_samples_split, min_samples_leaf, max_features |
| Decision Tree | max_depth (2–20), min_samples_split, min_samples_leaf, criterion (gini/entropy) |
| Naïve Bayes | var_smoothing (1e-12 to 1e-6, log scale) |
| Logistic Regression | C (1e-4 to 100, log), penalty (l1/l2), solver |
| Neural Network | hidden layers (1–4), neurons (32–256), dropout (0.1–0.5), lr (1e-5 to 1e-2), batch_size |

Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

#### [NEW] [valuation_regressor.py](file:///c:/projects/ML_Project/src/models/valuation_regressor.py)
**Task 2: Valuation Prediction (Regression)**

| Model | Key Hyperparameters (Optuna) |
|-------|------------------------------|
| Random Forest | n_estimators, max_depth, min_samples_split, max_features |
| Decision Tree | max_depth, min_samples_split, min_samples_leaf |
| Naïve Bayes | (not typical for regression — will use BayesianRidge as proxy) |
| Logistic Regression | → replaced with Ridge/Lasso Regression (alpha, l1_ratio) |
| Neural Network | architecture, lr, epochs, weight_decay |

> [!NOTE]
> Naïve Bayes and Logistic Regression are classification algorithms. For the regression task:
> - **Naïve Bayes → BayesianRidge** (Bayesian linear regression)
> - **Logistic Regression → Ridge/Lasso/ElasticNet** (regularized linear regression)
> This keeps the spirit of "6 different model families" while being mathematically appropriate.

Metrics: MAE, RMSE, R², MAPE

#### [NEW] [shark_recommender.py](file:///c:/projects/ML_Project/src/models/shark_recommender.py)
**Task 3: Shark Recommendation (Multi-Label Classification)**

Uses `OneVsRestClassifier` wrapper for sklearn models and multi-output sigmoid for neural network.

Target: 7 binary labels (one per shark: Namita, Vineeta, Anupam, Aman, Peyush, Ritesh, Amit)

| Model | Approach |
|-------|----------|
| Random Forest | `OneVsRestClassifier(RandomForestClassifier(...))` |
| Decision Tree | `OneVsRestClassifier(DecisionTreeClassifier(...))` |
| Naïve Bayes | `OneVsRestClassifier(GaussianNB(...))` |
| Logistic Regression | `OneVsRestClassifier(LogisticRegression(...))` |
| Neural Network | Multi-output with BCEWithLogitsLoss |

Metrics: Micro-F1, Macro-F1, Hamming Loss, Per-Shark Accuracy, Subset Accuracy

---

### Component 5: Optuna Tuning Engine

#### [NEW] [optuna_tuner.py](file:///c:/projects/ML_Project/src/tuning/optuna_tuner.py)
Unified tuning orchestrator:
- `tune_model(model_class, task, X_train, y_train, n_trials=100)` 
- Uses `StratifiedKFold` (classification) or `KFold` (regression) cross-validation inside objective
- Implements **MedianPruner** for neural network trials
- Stores study in SQLite DB (`optuna_studies/{task}_{model}.db`) for resumability
- Generates Optuna visualizations:
  - Parameter importance plot
  - Optimization history
  - Parallel coordinate plot
  - Contour plot
- Returns best trial params + retrained best model

---

### Component 6: Evaluation & Visualization

#### [NEW] [metrics.py](file:///c:/projects/ML_Project/src/evaluation/metrics.py)
- `classification_report_dict()` — extended sklearn classification report
- `regression_report_dict()` — MAE, RMSE, R², MAPE
- `multilabel_report_dict()` — per-label and aggregate multi-label metrics
- `model_comparison_table()` — comparative DataFrame of all models
- `cross_val_scores()` — stratified cross-validation with confidence intervals

#### [NEW] [visualization.py](file:///c:/projects/ML_Project/src/utils/visualization.py)
Reusable plotting functions:
- `plot_confusion_matrix()` — heatmap for classification
- `plot_roc_curve()` — ROC curves for all classification models
- `plot_feature_importance()` — bar chart for tree-based models
- `plot_actual_vs_predicted()` — scatter plot for regression
- `plot_residuals()` — residual analysis for regression
- `plot_multilabel_heatmap()` — per-shark prediction heatmap
- `plot_model_comparison()` — bar chart comparing all models on key metrics
- `plot_optuna_results()` — wrapper for Optuna's built-in visualizations
- All plots use a consistent theme (seaborn darkgrid, custom color palette)

#### [NEW] [helpers.py](file:///c:/projects/ML_Project/src/utils/helpers.py)
- `set_seed(seed)` — reproducibility
- `save_model(model, path)` / `load_model(path)` — joblib wrapper
- `log_results(results_dict, filepath)` — JSON logging
- `timer_decorator()` — training time measurement

---

### Component 7: Training Orchestrator

#### [NEW] [train.py](file:///c:/projects/ML_Project/train.py)
Main script that:
1. Loads and preprocesses data
2. For each task (deal, valuation, shark):
   - For each model family:
     - Runs Optuna hyperparameter tuning
     - Trains final model with best params
     - Evaluates on test set
     - Saves model artifact
3. Generates comparison reports
4. Saves all results to JSON

Can be run with CLI args: `python train.py --task deal --model rf --trials 50`

---

### Component 8: Streamlit Web Application

#### [NEW] [streamlit_app.py](file:///c:/projects/ML_Project/app/streamlit_app.py)
Interactive web interface with:

**Sidebar:**
- Input fields for startup details (industry, revenue, ask amount, equity, etc.)
- Model selection dropdowns (choose which trained model to use per task)

**Main Area — 3 Tabs:**
1. **Deal Prediction**: Shows probability gauge, feature importance, SHAP explanation
2. **Valuation Estimation**: Shows predicted valuation with confidence interval, comparison with similar deals
3. **Shark Recommendation**: Shows radar chart of per-shark probability, top-3 recommended sharks with investment style profiles

**Additional Pages:**
- **Model Performance Dashboard**: Compare all models' metrics in interactive charts
- **EDA Dashboard**: Key visualizations from the notebooks
- **About**: Project description, methodology, team info

---

## Open Questions

> [!IMPORTANT]
> 1. **Dataset download method**: Manual download or Kaggle API? (See "User Review Required" above)
> 2. **Neural Network framework**: I'm planning **PyTorch** for the neural networks. Would you prefer **Keras/TensorFlow** instead?
> 3. **Optuna trial count**: Planning 100 trials per model. Want more/fewer?
> 4. **Guest sharks**: The dataset has guest investors. Should we include them as a separate label in shark recommendation, or only focus on the 7 main sharks?
> 5. **Naïve Bayes / Logistic Regression for regression**: Are you okay with BayesianRidge and Ridge/Lasso as the regression equivalents? Or do you want me to bin valuation into categories and keep it as classification?

---

## Verification Plan

### Automated Tests
- Run `python train.py --task deal --model dt --trials 5` (quick smoke test with Decision Tree, 5 trials)
- Verify all model artifacts are saved to `models/` directory
- Verify Optuna studies persist in `optuna_studies/`
- Run `streamlit run app/streamlit_app.py` and verify all 3 prediction tasks work

### Manual Verification
- Review EDA notebooks for data quality insights
- Check model comparison metrics look reasonable (deal prediction accuracy should be >65%, valuation R² > 0.3)
- Test Streamlit app with sample inputs
- Verify plots render correctly
