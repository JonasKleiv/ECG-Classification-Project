import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

# ==============================================================================
# 1. Sample Weighting and Missing Value Helpers
# ==============================================================================

def compute_clinical_weights(y_encoded, encoder, clinical_multipliers=None):
    """
    Compute balanced sample weights and optionally multiply by clinical class priorities.
    """
    weights = compute_sample_weight(class_weight="balanced", y=y_encoded)
    
    if clinical_multipliers is not None:
        weights = np.copy(weights)
        for class_name, multiplier in clinical_multipliers.items():
            class_idx = encoder.transform([class_name])[0]
            weights[y_encoded == class_idx] *= multiplier

    return weights

def compute_multilabel_class_weights(y_binarized, class_names, clinical_multipliers=None):
    """
    Compute per-output class_weight dictionaries
    """
    class_weights = []
    n_samples = y_binarized.shape[0]

    for idx, name in enumerate(class_names):
        pos = y_binarized[:, idx].sum()
        neg = n_samples - pos
        multiplier = clinical_multipliers.get(name, 1.0) if clinical_multipliers else 1.0
        
        w_pos = (n_samples / (2.0 * pos)) * multiplier
        w_neg = n_samples / (2.0 * neg)
        class_weights.append({0: w_neg, 1: w_pos})

    return class_weights

def clean_and_impute_features(X_df, y_encoded, strat_folds, min_valid_ratio=0.8, strategy="mean"):
    """
    For models that do not natively handle NaNs (KNN, Decision Tree, SVM, Random Forest):
    drop rows missing more than (1 - min_valid_ratio) features and impute remaining NaNs.
    """
    # Drop rows that have fewer valid features than the minimum required
    df_clean = X_df.copy().reset_index(drop=True)
    min_valid_features = int(df_clean.shape[1] * min_valid_ratio)
    df_clean = df_clean.dropna(thresh=min_valid_features)
    
    # Impute remaining missing values using the specified strategy
    valid_idx = df_clean.index.to_numpy()
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(df_clean)
    
    # Ensure the imputed features have the same number of rows as the cleaned labels and strat_folds
    y_clean = np.asarray(y_encoded)[valid_idx]
    folds_clean = np.asarray(strat_folds)[valid_idx]
    
    return X_imputed, y_clean, folds_clean

# ==============================================================================
# 2. Excel Row and Classification Report Evaluation
# ==============================================================================
def generate_excel_row(y_true, y_pred, feature_count, class_names, excel_classes=None):
    """Format accuracy and per-class Precision/Recall/F1 as a tab-separated row for Excel."""
    if excel_classes is None:
        excel_classes = ["NORM", "MI", "STTC", "HYP", "CD"]
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    output_row = f"{feature_count}\t{report['accuracy']:.4f}"
    
    for cls in excel_classes:
        if cls in report:
            output_row += (
                f"\t{report[cls]['precision']:.4f}"
                f"\t{report[cls]['recall']:.4f}"
                f"\t{report[cls]['f1-score']:.4f}"
            )
        else:
            output_row += "\t\t\t"
    
    return output_row

def evaluate_single_label_cv(
    model,
    X,
    y_encoded,
    strat_folds,
    class_names,
    title="Model Evaluation",
    sample_weight=None,
    n_jobs=-3,
    print_excel_row=True,
):
    """
    Evaluate single-label model using official PTB-XL recommended benchmark splits:
    - Train on Folds 1-9 (strat_folds <= 9)
    - Test on Fold 10 (strat_folds == 10, human cardiologist-validated gold standard).
    Runs only once instead of k-fold CV.
    """
    folds_arr = np.asarray(strat_folds)
    train_mask = (folds_arr <= 9)
    test_mask = (folds_arr == 10)

    X_train = X[train_mask] if isinstance(X, np.ndarray) else X.iloc[train_mask]
    y_train = y_encoded[train_mask]
    w_train = sample_weight[train_mask] if sample_weight is not None else None

    X_test = X[test_mask] if isinstance(X, np.ndarray) else X.iloc[test_mask]
    y_test = y_encoded[test_mask]

    fit_kwargs = {}
    if w_train is not None:
        fit_kwargs["sample_weight"] = w_train

    # Fit model once on training folds 1-9
    model.fit(X_train, y_train, **fit_kwargs)

    # Predict once on gold-standard test fold 10
    y_pred_test = model.predict(X_test)

    print(f"\n------ Official PTB-XL Test Fold 10 Classification Report ({title}) ------")
    print(classification_report(y_test, y_pred_test, target_names=class_names))
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred_test) * 100:.2f}%")
    print(f"Macro F1-Score:   {f1_score(y_test, y_pred_test, average='macro'):.4f}")
    print(f"Feature Count:    {X.shape[1]}")
    
    if print_excel_row:
        row = generate_excel_row(y_test, y_pred_test, X.shape[1], class_names)
        print("\r\nExcel Row:\n" + row)
    
    return y_pred_test

def evaluate_multi_label_cv(
    model,
    X,
    y_binarized,
    strat_folds,
    class_names,
    title="Multi-Label Evaluation",
    sample_weight=None,
    n_jobs=-3,
):
    """
    Evaluate multi-label model using official PTB-XL recommended benchmark splits:
    - Train on Folds 1-9 (strat_folds <= 9)
    - Test on Fold 10 (strat_folds == 10, human cardiologist-validated gold standard).
    Runs only once instead of k-fold CV.
    """
    folds_arr = np.asarray(strat_folds)
    train_mask = (folds_arr <= 9)
    test_mask = (folds_arr == 10)

    X_train = X[train_mask] if isinstance(X, np.ndarray) else X.iloc[train_mask]
    y_train = y_binarized[train_mask]
    w_train = sample_weight[train_mask] if sample_weight is not None else None

    X_test = X[test_mask] if isinstance(X, np.ndarray) else X.iloc[test_mask]
    y_test = y_binarized[test_mask]

    fit_kwargs = {}
    if w_train is not None:
        fit_kwargs["sample_weight"] = w_train

    # Fit model once on training folds 1-9
    model.fit(X_train, y_train, **fit_kwargs)

    # Predict once on gold-standard test fold 10
    y_pred_test = model.predict(X_test)

    print(f"\n====== Official PTB-XL Test Fold 10 Multi-Label Classification Report ({title}) ======")
    
    for idx, class_name in enumerate(class_names):
        print(f"\nClass: {class_name}")
        print(
            classification_report(
                y_test[:, idx],
                y_pred_test[:, idx],
                target_names=[f"Not {class_name}", class_name],
                zero_division=0,
            )
        )
    
    macro_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    h_loss = hamming_loss(y_test, y_pred_test)
    
    print("------------------------------------------------------------")
    print(f"Multi-Label Macro F1-Score: {macro_f1:.4f}")
    print(f"Overall Hamming Loss:       {h_loss:.4f}")
    
    return y_pred_test

def evaluate_auroc(
        model,
        X,
        y,
        strat_folds,
        class_names,
        multi_label=False,
        title="AUROC Evaluation",
):
    """
    Compute AUROC on official PTB-XL test fold 10 using an already-fitted model.
    X, y and strat_folds must be row-aligned (use the folds returned by
    clean_and_impute_features when X was cleaned).
    """
    test_mask = np.asarray(strat_folds) == 10
    X_test = X.iloc[test_mask] if hasattr(X, "iloc") else X[test_mask]
    y_test = y[test_mask]

    if multi_label:
        proba = np.column_stack([p[:, 1] for p in model.predict_proba(X_test)])
        per_class = [roc_auc_score(y_test[:, i], proba[:, i]) for i in range(len(class_names))]
        macro = roc_auc_score(y_test, proba, average="macro")
    else:
        proba = model.predict_proba(X_test)
        per_class = [roc_auc_score(y_test == i, proba[:, i]) for i in range(len(class_names))]
        macro = roc_auc_score(y_test, proba, average="macro", multi_class="ovr")

    print(f"\n------ AUROC on Test Fold 10 ({title}) ------")
    for name, auc in zip(class_names, per_class):
        print(f"AUROC {name}: {auc:.4f}")
    print(f"Macro AUROC: {macro:.4f}")

# ==============================================================================
# 3. Feature Selection (RFECV, Collinearity), Hyperparameter Tuning and Importance
# ==============================================================================

def run_rfecv_selection(
    base_estimator,
    X_df,
    y_encoded,
    strat_folds,
    output_csv_path,
    step=5,
    n_jobs=-3,
    force_reload=False,
):
    """
    Run Recursive Feature Elimination with Cross-Validation (RFECV) using PTB-XL's official folds:
    - Trains on Folds 1-8
    - Validates feature elimination performance on Fold 9 (cardiologist-validated validation set)
    - Zero data leakage from Fold 10 (test set).
    """
    if os.path.exists(output_csv_path) and not force_reload:
        print(f"Loading cached RFE features from: {output_csv_path}")
        return pd.read_csv(output_csv_path)

    folds_arr = np.asarray(strat_folds)
    rfe_mask = (folds_arr <= 9)
    X_rfe_train = X_df[rfe_mask] if isinstance(X_df, np.ndarray) else X_df.iloc[rfe_mask]
    y_rfe_train = y_encoded[rfe_mask]

    # PredefinedSplit: -1 = train (folds 1-8), 0 = validation (fold 9)
    test_fold_rfe = np.where(folds_arr[rfe_mask] == 9, 0, -1)
    cv_rfe = PredefinedSplit(test_fold=test_fold_rfe)

    f1_macro_scorer = make_scorer(f1_score, average="macro")

    rfecv = RFECV(
        estimator=clone(base_estimator),
        step=step,
        cv=cv_rfe,
        scoring=f1_macro_scorer,
        n_jobs=n_jobs,
    )

    print(f"Starting RFECV Feature Selection on Folds 1-8 with Fold 9 Val (step={step})...")
    rfecv.fit(X_rfe_train, y_rfe_train)
    selected_cols = X_df.columns[rfecv.support_]

    # Return full matrix with selected columns
    rfe_df = X_df[selected_cols].copy()
    rfe_df.to_csv(output_csv_path, index=False)
    print(f"Saved {rfe_df.shape[1]} RFE-selected features to: {output_csv_path}")
    
    return rfe_df

def filter_collinear_features(
    X_df,
    output_csv_path=None,
    threshold=0.90,
    force_reload=False,
):
    """
    Drop one feature from every pair whose absolute Pearson correlation exceeds
    `threshold`, or load the filtered feature matrix from cache if available.
    """
    if output_csv_path and os.path.exists(output_csv_path) and not force_reload:
        print(f"Loading cached collinearity-filtered features from: {output_csv_path}")
        return pd.read_csv(output_csv_path)

    corr_matrix = X_df.corr(method="pearson").abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]

    X_uncorrelated = X_df.drop(columns=cols_to_drop)

    print(f"Original features:              {X_df.shape[1]}")
    print(f"Dropped features (|r| > {threshold:.2f}): {len(cols_to_drop)}")
    print(f"Remaining features:             {X_uncorrelated.shape[1]}")

    if output_csv_path:
        X_uncorrelated.to_csv(output_csv_path, index=False)
        print(f"Saved {X_uncorrelated.shape[1]} uncorrelated features to: {output_csv_path}")

    return X_uncorrelated

def tune_and_evaluate_model(
    base_estimator,
    param_distributions,
    X_data,
    y_encoded,
    strat_folds,
    class_names,
    label_name="Tuned Model",
    n_iter=20,
    sample_weight=None,
    random_state=42,
    n_jobs=-3,
):
    """
    Tune hyperparameters on folds 1-8 (train) and fold 9 (cardiologist-validated val set).
    Zero data leakage from fold 10 (test set).
    Refits best parameters on folds 1-9, then evaluates once on gold-standard fold 10.
    """
    print(f"================ {label_name} ({X_data.shape[1]} features) ================")
    
    folds_arr = np.asarray(strat_folds)
    tune_mask = (folds_arr <= 9)
    X_tune = X_data[tune_mask] if isinstance(X_data, np.ndarray) else X_data.iloc[tune_mask].reset_index(drop=True)
    y_tune = y_encoded[tune_mask]
    weights_tune = sample_weight[tune_mask] if sample_weight is not None else None

    # PredefinedSplit: -1 = train (folds 1-8), 0 = validation (fold 9)
    test_fold_tune = np.where(folds_arr[tune_mask] == 9, 0, -1)
    cv_tune = PredefinedSplit(test_fold=test_fold_tune)

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv_tune,
        verbose=1,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )

    fit_kwargs = {}
    if weights_tune is not None:
        fit_kwargs["sample_weight"] = weights_tune

    search.fit(X_tune, y_tune, **fit_kwargs)

    print(f"\nBest Tuning Macro F1 (Fold 9 Val) ({label_name}): {search.best_score_:.4f}")
    print(f"Best Parameters ({label_name}):\n{search.best_params_}")

    best_model = search.best_estimator_
    
    # Evaluate refit best model once on Fold 10
    evaluate_single_label_cv(
        model=best_model,
        X=X_data,
        y_encoded=y_encoded,
        strat_folds=strat_folds,
        class_names=class_names,
        title=label_name,
        sample_weight=sample_weight,
        n_jobs=n_jobs,
    )

    return best_model, search.best_params_

# ==============================================================================
# 4. Visualization Helpers (Confusion Matrix, Feature Importance and Collinearity)
# ==============================================================================
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Plot a formatted seaborn confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.set_theme(style="white")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Number of Patients"},
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Diagnosis", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Diagnosis", fontsize=12, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_top_feature_importances(model, feature_names=None, top_n=20, title="Top Feature Importances"):
    """Extract and plot the top N most important features from a fitted tree-based model."""
    if feature_names is None:
        feature_names = model.feature_names_in_

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 6))
    top_df = importance_df.head(top_n).iloc[::-1]
    plt.barh(top_df["Feature"], top_df["Importance"], color="steelblue")
    plt.xlabel("Feature Importance (Gain)")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return importance_df

def plot_correlation_matrix(
    X_df,
    title="Feature Correlation Matrix",
    lead=None,
    features=None,
    show_labels=None,
    annot=False,
    pqrst_boundary=86,
):
    """
    Plot a Pearson correlation heatmap.
    - If lead is given, plots only that lead's PQRST + DWT features (27 features).
    - If features (list of column names) is given, plots only those features.
    - Otherwise plots the full matrix.
    """
    df_sub = X_df
    if lead is not None:
        p1, p2 = f"lead{lead}_", f"Lead{lead}_"
        cols = [c for c in X_df.columns if c.startswith(p1) or c.startswith(p2)]
        df_sub = X_df[cols]
        pqrst_boundary = sum(c.startswith(p1) for c in cols)
        if title == "Feature Correlation Matrix":
            title = f"Lead {lead} Feature Correlation Matrix (PQRST vs. DWT)"
    elif features is not None:
        df_sub = X_df[list(features)]
        pqrst_boundary = None

    if show_labels is None:
        show_labels = df_sub.shape[1] <= 40

    corr = df_sub.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set_theme(style="white")
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        annot=annot,
        fmt=".2f",
        xticklabels=corr.columns if show_labels else False,
        yticklabels=corr.columns if show_labels else False,
        cbar_kws={"label": "Pearson Correlation (r)", "shrink": 0.8},
        ax=ax,
    )

    if show_labels:
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        if pqrst_boundary and pqrst_boundary < df_sub.shape[1]:
            ax.axhline(pqrst_boundary, color="black", linestyle="--", linewidth=1.2)
            ax.axvline(pqrst_boundary, color="black", linestyle="--", linewidth=1.2)
    elif pqrst_boundary and pqrst_boundary < df_sub.shape[1]:
        ax.axhline(pqrst_boundary, color="black", linestyle="--", linewidth=1.2)
        ax.axvline(pqrst_boundary, color="black", linestyle="--", linewidth=1.2)
        ax.set_xticks([pqrst_boundary / 2, pqrst_boundary + (df_sub.shape[1] - pqrst_boundary) / 2])
        ax.set_xticklabels(["PQRST Features (1–86)", f"DWT Features (87–{df_sub.shape[1]})"], fontsize=11, fontweight="bold")
        ax.set_yticks([pqrst_boundary / 2, pqrst_boundary + (df_sub.shape[1] - pqrst_boundary) / 2])
        ax.set_yticklabels(["PQRST Features (1–86)", f"DWT Features (87–{df_sub.shape[1]})"], fontsize=11, fontweight="bold", rotation=90, va="center")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.show()

    return corr
