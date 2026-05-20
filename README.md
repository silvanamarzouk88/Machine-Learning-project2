 FIFA Player Valuation & Performance Tier Prediction

End-to-end ML pipeline on FIFA player data covering regression (market value prediction) and multi-class classification (performance tier), with ensemble methods and a unified inference system.

## Dataset

`Fifa.csv` — FIFA player records with attributes including Age, Overall Rating, Future Potential, Total Stats Score, Position, Country, and Team. No missing values were found in the dataset.

## Targets

- **Regression:** `Value Per M$` — player market value in millions
- **Classification:** `Tier` — 4-class performance label (Low / Mid / High / Elite) derived from quartile splits of `Overall_Rating`

## Pipeline

### 1. Exploratory Data Analysis
- Distribution analysis of `Value Per M$` (right-skewed, mean > median)
- Correlation heatmap across all numerical features
- Top features correlated with market value: Overall Rating and Future Potential dominate
- Average Overall Rating per position visualized as a bar chart

### 2. Target Engineering
`Tier` is created using `pd.qcut` on `Overall_Rating` with q=4, producing approximately balanced classes. `Overall_Rating` was chosen over `Total_Stats Score` for its higher correlation with the target and stronger domain interpretability.

| Tier  | Overall Rating Range |
|-------|---------------------|
| Low   | 36 – 58             |
| Mid   | 58 – 63             |
| High  | 63 – 68             |
| Elite | 68 – 91             |

### 3. Preprocessing
All steps fit on train only and applied to test without refitting.

- 80/20 train/test split before any preprocessing
- `Name` dropped (identifier); `Overall_Rating` excluded from classification features to prevent data leakage
- **Encoding:** One-Hot Encoding for Position; Frequency Encoding for Country and Team; Label Encoding for Tier
- **Outlier Capping:** IQR method on Age, Future Potential, Total Stats Score (and Overall Rating for regression only)
- **Scaling:** StandardScaler on numerical features

### 4. Regression Models — Predict Value Per M$

**Polynomial Regression with Regularization**

Degrees 1–4 tested. Degree 4 selected (Test R² = 0.8773). Ridge and Lasso regularization applied via alpha sweep (log-spaced 0.001–1000). Lasso eliminated 19 of 85 features, primarily high-degree interaction terms and rare position dummies.

| Model | Test R² | Test RMSE |
|---|---|---|
| Linear (Degree 1) | 0.3162 | 6.5126 |
| Polynomial Degree 4 | 0.8773 | 2.7593 |
| Ridge (α=59.64) | 0.8855 | 2.6648 |
| Lasso (α=0.0023) | 0.8856 | 2.6641 |

**KNN Regressor** — GridSearchCV over k=1–30 and Euclidean/Manhattan distance. Best: k=6, Manhattan. Test R²=0.8622.

**SVR** — GridSearchCV over C, kernel, gamma, epsilon. Best: C=100, RBF kernel. Mild overfitting observed; gap narrows with more data.

**Random Forest Regressor** — GridSearchCV; best params: n_estimators=300, max_depth=15, min_samples_split=10. Test R²=0.9258. Feature importance: Overall Rating (69%), Future Potential (23%), Age (5%).

### 5. Classification Models — Predict Performance Tier

**Logistic Regression** — C sweep (10⁻³ to 10³); best C chosen by test accuracy. L1 vs L2 compared; L2 retained as it performed better, indicating all features contribute.

**Naive Bayes** — Three variants tested on appropriate feature subsets.

| Model | Features Used | Test Accuracy |
|---|---|---|
| GaussianNB | Continuous numerical | 71% |
| BernoulliNB | OHE binary features | 30% |
| ComplementNB | OHE binary features | 30% |

GaussianNB is the most appropriate variant given the continuous numerical features.

**KNN Classifier** — GridSearchCV; best: k=18, Manhattan. Test Accuracy=0.8447.

**SVC** — GridSearchCV; best: C=100, RBF kernel. Test Accuracy=0.8566, good generalization with minimal train/test gap.

**Random Forest Classifier** — GridSearchCV with aggressive regularization (max_samples=0.8, shallow max_depth). Test Accuracy=0.8338. Feature importance: Total Stats Score (37%), Future Potential (29%), Age (24%).

### 6. Cross-Validation
- Regression: 5-Fold KFold on Lasso and RF Regressor
- Classification: Stratified 5-Fold on Logistic Regression, GaussianNB, and RF Classifier
- Logistic Regression outperforms GaussianNB in both mean accuracy and stability

### 7. Ensemble Methods

**Voting and Stacking on Classification** (SVM + RF + KNN):

| Method | Test Accuracy |
|---|---|
| Voting | 0.8584 |
| Stacking | 0.8594 |

**Voting and Stacking on Regression** (RF + Lasso + KNN):

| Method | Test R² |
|---|---|
| Voting | 0.8223 |
| Stacking | 0.9267 |

Stacking significantly outperforms Voting in regression because the meta-model learns optimal combination weights rather than simple averaging.

### 8. Unified Inference System
A `DataPreparationTransformer` (sklearn-compatible) encapsulates all preprocessing logic into a reusable pipeline. Two end-to-end pipelines are built:

- **Regression Pipeline:** `DataPreparationTransformer` → `StackingRegressor` (RF + Lasso + KNN → Linear meta-model)
- **Classification Pipeline:** `DataPreparationTransformer` → `StackingClassifier` (RF + KNN + SVC → Logistic meta-model)

Both pipelines accept raw, unprocessed player data, making them production-ready and leak-free. An interactive prompt (`predict_player_interactive`) allows entering player details and receiving both predictions simultaneously.

### 9. System Comparison vs Assignment 2 Baselines

| Task | A2 Baseline | Best System | Improvement |
|---|---|---|---|
| Regression R² | 0.3162 | 0.9267 | +61% |
| Classification Accuracy | 0.8155 | 0.8609 | +4.5% |

### 10. Stability Assessment

| Task | Model | Mean CV Score | Std |
|---|---|---|---|
| Regression | Stacking | ~0.92 R² | low |
| Classification | Stacking | ~0.86 Acc | low |

Low standard deviation across 5 folds confirms stable generalization.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Run notebook cells in order. The interactive prediction system at the end prompts for player details and outputs both market value and performance tier simultaneously.
