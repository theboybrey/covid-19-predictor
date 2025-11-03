# Ghana COVID-19 ML Project

Dataset source: https://www.kaggle.com/datasets/kuleafenujoachim/ghana-covid19-cases-dataset

Project goal
-- Use the Ghana COVID-19 dataset to identify a real-life problem and apply machine learning to analyze and predict outcomes (example: case counts trend forecasting or risk classification).

Proposed outline

1. Acquire dataset (Kaggle)
2. Inspect and identify the prediction or analysis problem
3. Clean and preprocess data (`src/preprocess.py`)
4. Feature engineering (`src/features.py`)
5. Train models (`src/train_model.py`)
6. Evaluate and present findings (`src/evaluate.py`, `reports/`)

How to get the data (Kaggle API)

Prerequisites
- Create a Kaggle account and generate an API token at https://www.kaggle.com/me/account
- Place the downloaded `kaggle.json` file into `~/.kaggle/kaggle.json` and ensure it is readable only by you (chmod 600 ~/.kaggle/kaggle.json)
- Or set the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Automatic download (recommended)

Run the ingestion script which uses the Kaggle API:

```bash
# from project root
python src/data_ingest.py
```

This will download and extract the dataset into `data/raw/`.

Manual download

If you prefer not to use the Kaggle API, download the dataset manually from the Kaggle link above and place the files inside `data/raw/`.

Quick setup

1. Create and activate a Python virtual environment (venv or conda).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download data (automatic or manual as above).
4. Run preprocessing and split:

```bash
python src/preprocess.py
```

5. Run EDA in `notebooks/01-exploration.ipynb` or run training:

```bash
python src/train_model.py
```

Structure

- data/raw/     # raw downloaded files
- data/processed/ # cleaned and split data
- notebooks/    # EDA and analysis notebooks
- src/          # preprocessing, features, model training, evaluation
- reports/figures/ # saved figures and report assets

Notes
- The provided `src/` scripts contain generic placeholders; they will be adapted to the Ghana dataset (date parsing, case columns, region breakdown) once the raw data is inspected.

