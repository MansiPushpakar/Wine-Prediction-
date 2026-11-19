# Wine Quality Prediction — Improved Streamlit App

A Streamlit deployment of the original **Red Wine Quality binary classification** project.

## What this project predicts

The original notebook uses the UCI Red Wine Quality dataset (1,599 records, 11 physicochemical input features and a numeric `quality` score). It converts the target into two classes:

- `1` — Good quality (`quality >= 7`)
- `0` — Not good quality (`quality < 7`)

This project preserves that target rule, Random Forest model family, 80:20 train-test split, and `random_state=3` split setting.

## Files

```text
.
├── app.py
├── model.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── tests/
    └── test_model.py
```

## Dataset

The app first checks for `winequality-red.csv` in the project root. If it is not present, it downloads the public red-wine CSV from the Hugging Face dataset mirror at startup. The CSV is semicolon-separated.

Dataset citation:
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality*. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T

For an offline deployment, download `winequality-red.csv` from the UCI dataset page and place it beside `app.py`.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Run tests

```bash
pytest -q
```

The tests use a small synthetic fixture and do not need internet access or the production CSV.

## Deploy on Streamlit Community Cloud

1. Upload the project files and folders to your GitHub repository.
2. Open https://share.streamlit.io/
3. Select the repository and branch.
4. Set **Main file path** to `app.py`.
5. Deploy.

No API key or Streamlit secrets are required.

## Evaluation and reproducibility

The app calculates accuracy, precision, recall and F1 on the held-out test split at runtime. The notebook's recorded accuracy (93.125%) came from its original run using an unseeded `RandomForestClassifier()`; this deployment fixes the estimator seed for reproducibility, so its measured score may differ. Do not advertise the notebook's 93.125% as the app's current score unless the deployed app reports the same value.

## Quality and safety notes

- All required columns, missing values, numeric values and finite prediction inputs are validated.
- The model is cached by Streamlit to avoid retraining on every interaction.
- The UI includes labels, help text, accessible contrast and responsive Streamlit columns.
- This is an educational ML demonstration, not a certified wine-testing system.
