import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def load_data(path="data/processed/train.csv"):
    return pd.read_csv(path)


def train_baselines(X, y):
    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gbr": GradientBoostingRegressor(random_state=42)
    }
    if _HAS_XGB:
        models['xgboost'] = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)

    results = {}
    for name, model in models.items():
        # use negative MAE for cross_val_score, convert later
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        mae_scores = -scores
        results[name] = {"mae_mean": float(mae_scores.mean()), "mae_std": float(mae_scores.std())}
    return results, models


def grid_search_rf(X, y):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    gs = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    gs.fit(X, y)
    os.makedirs('models', exist_ok=True)
    dump(gs.best_estimator_, "models/rf_best.joblib")
    return gs


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    # if train.csv doesn't exist, attempt to create supervised data from timeseries
    if not os.path.exists("data/processed/train.csv"):
        try:
            from src.features import create_supervised_from_timeseries
        except Exception:
            # try relative import fallback
            from features import create_supervised_from_timeseries
        print("train.csv not found — creating supervised dataset from timeseries...")
        create_supervised_from_timeseries()

    df = load_data()
    target_col = 'target'
    if target_col not in df.columns:
        raise KeyError(f"Expected target column '{target_col}' in data/processed/train.csv")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    results, models = train_baselines(X, y)
    print("Baseline cross-validated MAE:", results)
    gs = grid_search_rf(X, y)
    print("Best RF params:", gs.best_params_)
