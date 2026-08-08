# Challenges Faced

This document summarizes the key challenges encountered while building the Automobile Price Prediction model, and how each was addressed.

## 1. Messy Raw Data / Missing Column Headers

The raw CSV (`auto_imports.csv`) had no header row and used `"?"` to represent missing values instead of a standard null marker. This meant the first naive `pd.read_csv()` load produced meaningless column names (`0, 1, 2, ...`) and treated missing values as literal strings.

**Fix:** Re-loaded the data with an explicit `column_names` list and `na_values="?"` so pandas correctly parsed the headers and recognized missing values as `NaN`.

## 2. Word-Based Categorical Values Mixed with Numeric Meaning

Columns like `num_of_doors` and `num_of_cylinders` stored counts as English words (`"four"`, `"six"`, `"twelve"`) instead of numbers, so they were initially treated as free-text categorical features rather than ordinal/numeric ones.

**Fix:** Built a `word_to_num` mapping dictionary and applied it early — before splitting into `X`/`y` — so these columns became true numeric features usable by the model.

## 3. Column List Staleness Causing Pipeline Errors

Several `numerical_col` / `categorical_col` variables were computed early during EDA (on the full `df`, including the `price` target). Later, when building the `ColumnTransformer`, these stale lists were referenced instead of the freshly computed feature lists derived from `X` (after dropping `price`). This caused a `ValueError: A given column is not a column of the dataframe`, since `price` no longer existed in `X_train`.

**Fix:** Recomputed `numeric_features` / `categorical_features` from `X` right before building the preprocessor, and made sure the `ColumnTransformer` referenced those instead of the outdated EDA-time lists.

## 4. Outliers Skewing the Target and Several Features

`price`, `horsepower`, and a few other numeric columns had a noticeable right skew with some extreme outliers, which could disproportionately influence linear models.

**Fix:** Applied IQR-based clipping (1.5×IQR bounds) across numerical columns to cap extreme values without discarding rows, preserving sample size while reducing the influence of outliers.

## 5. Encoding Multiple Types of Categorical Data Correctly

The dataset mixed binary categoricals (`fuel_type`, `aspiration`, `engine_location`), nominal categoricals (`make`, `body_style`, `drive_wheels`), and word-based ordinal counts — each needing a different encoding strategy. A single blanket encoding approach would have either lost ordinal meaning or introduced unnecessary dimensionality.

**Fix:** Split features into distinct groups and built separate pipelines (imputers + encoders) for each, combined via a single `ColumnTransformer` so every column type was handled appropriately.

## 6. Comparing Many Models Fairly

Evaluating ten different regressors (linear, tree-based, boosting, SVR, KNN) required a consistent evaluation protocol so results were comparable rather than an apples-to-oranges comparison.

**Fix:** Wrapped every model in the same `Pipeline` (shared `preprocessor` + model step) and evaluated all of them with identical `train_test_split`, `KFold` cross-validation, and metrics (R², MAE, RMSE).

## 7. Hyperparameter Tuning Cost

Grid-searching multiple hyperparameters across several tree/boosting models was computationally expensive, especially with cross-validation multiplying the number of fits.

**Fix:** Scoped `param_grids` to a small, high-impact set of hyperparameters per model rather than exhaustively searching every possible combination.

## 8. Model Persistence and Reproducibility

Saving the trained pipeline with `pickle` raised the risk of version incompatibility — a pickled scikit-learn/XGBoost object may fail to load in an environment with different library versions.

**Fix:** Saved the entire fitted `Pipeline` (preprocessing + model) as a single artifact so it's self-contained, and noted the importance of pinning `scikit-learn`/`xgboost` versions between save and load environments.

## Key Takeaways

- Always recompute derived variables (like column lists) after upstream transformations — stale references are an easy source of silent bugs.
- Apply feature transformations before the train/test split only when they don't leak target information; apply anything target-dependent after splitting.
- A consistent pipeline design across all candidate models makes comparison fair and results trustworthy.