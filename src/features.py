import pandas as pd
import os

def add_age_groups(df: pd.DataFrame, age_col="age"):
    # create bins
    bins = [0, 18, 35, 50, 65, 120]
    labels = ["child","young_adult","adult","mid_age","senior"]
    df['age_group'] = pd.cut(df[age_col], bins=bins, labels=labels)
    df = pd.get_dummies(df, columns=['age_group'], drop_first=True)
    return df

def symptom_count(df: pd.DataFrame, symptom_cols: list):
    df['symptom_count'] = df[symptom_cols].sum(axis=1)
    return df


def create_supervised_from_timeseries(ts_path: str = "data/processed/cases_timeseries.csv",
                                      lags: int = 14,
                                      rolling_windows: list = [7, 14],
                                      horizon: int = 1,
                                      test_days: int = 14,
                                      out_dir: str = "data/processed") -> pd.DataFrame:
    """Read a timeseries CSV (date index, daily_cases) and produce supervised
    train/test CSVs for regression.

    Args:
        ts_path: path to the timeseries CSV (expects columns 'daily_cases', optional 'cumulative_cases')
        lags: number of lag features to create (lag_1 ... lag_n)
        rolling_windows: list of window sizes to compute rolling means
        horizon: prediction horizon in days (1 => predict next-day value)
        test_days: number of most recent days to reserve for test
        out_dir: directory to save train.csv and test.csv

    Returns:
        the full supervised DataFrame (with train/test not split)
    """
    df = pd.read_csv(ts_path, parse_dates=[0], index_col=0)
    if 'daily_cases' not in df.columns:
        raise KeyError("timeseries file must contain a 'daily_cases' column")

    s = df['daily_cases'].rename('y')
    X = pd.DataFrame(index=s.index)

    # lag features
    for i in range(1, lags + 1):
        X[f'lag_{i}'] = s.shift(i)

    # rolling features
    for w in rolling_windows:
        X[f'roll_mean_{w}'] = s.shift(1).rolling(window=w, min_periods=1).mean()
        X[f'roll_std_{w}'] = s.shift(1).rolling(window=w, min_periods=1).std().fillna(0)

    # growth / pct change
    X['pct_change_1'] = s.pct_change().shift(1).fillna(0)

    # target at horizon
    y = s.shift(-horizon)

    supervised = pd.concat([X, y.rename('target')], axis=1).dropna()

    # split train/test by time (last test_days as test)
    if test_days >= 1:
        train = supervised.iloc[:-test_days]
        test = supervised.iloc[-test_days:]
    else:
        train = supervised
        test = supervised.iloc[0:0]

    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    print(f"Saved supervised train/test to {out_dir}/train.csv and {out_dir}/test.csv")
    return supervised
