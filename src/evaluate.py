import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(model, X_test, y_test, model_name="model"):
    y_pred = model.predict(X_test)
    print(f"=== {model_name} Classification Report ===")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"reports/figures/{model_name}_confusion_matrix.png")
    plt.clf()

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr)
        plt.title(f"{model_name} ROC Curve (AUC={auc:.3f})")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(f"reports/figures/{model_name}_roc.png")
        plt.clf()

def feature_importance(model, feature_names, model_name="model"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)
        fi.tail(15).plot(kind="barh")
        plt.title(f"{model_name} Feature Importances")
        plt.tight_layout()
        plt.savefig(f"reports/figures/{model_name}_feature_importances.png")
        plt.clf()


def evaluate_regression(model, X_test, y_test, model_name="model"):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"=== {model_name} Regression Report ===")
    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f}")

    # plot predicted vs actual
    plt.figure(figsize=(8, 5))
    plt.plot(y_test.values, label='actual')
    plt.plot(y_pred, label='predicted')
    plt.legend()
    plt.title(f"{model_name} Predicted vs Actual (MAE={mae:.3f})")
    os_path = f"reports/figures/{model_name}_pred_vs_actual.png"
    import os
    os.makedirs(os.path.dirname(os_path), exist_ok=True)
    plt.savefig(os_path)
    plt.clf()
    # residuals
    resid = y_test.values - y_pred
    plt.hist(resid, bins=30)
    plt.title(f"{model_name} Residuals")
    plt.xlabel('error')
    plt.ylabel('count')
    plt.savefig(f"reports/figures/{model_name}_residuals.png")
    plt.clf()
    return {"mae": float(mae), "rmse": float(rmse)}
