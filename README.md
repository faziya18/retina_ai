# Retina AI

A consolidated project for OCT and diabetic retinopathy classification with Streamlit apps, training scripts, and evaluation tools.

## Project Structure

```
project-root/
├── assets/
├── data/
├── models/
├── runs/
├── pyproject.toml
├── README.md
├── requirements.txt
├── app.db
├── dr_resnet50.pth
├── oct_resnet18_c8.pth
├── retina_ai/
│   ├── __init__.py
│   ├── apps/
│   │   ├── __init__.py
│   │   ├── oct_screening.py
│   │   ├── oct_screening_basic.py
│   │   └── secure_dr_oct.py
│   └── scripts/
│       ├── __init__.py
│       ├── evaluate.py
│       ├── test_inference.py
│       ├── train_fundus.py
│       └── train_oct8.py
└── ...
```

## Installation

Install the dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Alternatively, install the package locally:

```bash
python -m pip install -e .
```

## Usage

### Run the OCT screening Streamlit app

```bash
streamlit run -m retina_ai.apps.oct_screening
```

### Run the secure DR + OCT Streamlit app

```bash
streamlit run -m retina_ai.apps.secure_dr_oct
```

### Run the legacy OCT app

```bash
streamlit run -m retina_ai.apps.oct_screening_basic
```

### Train the fundus DR model

```bash
python -m retina_ai.scripts.train_fundus
```

### Train the OCT 8-class model

```bash
python -m retina_ai.scripts.train_oct8 --data path/to/dataset --out models/oct8
```

### Evaluate a model

```bash
python -m retina_ai.scripts.evaluate
```

### Run single-image inference

```bash
python -m retina_ai.scripts.test_inference
```

## Notes

- `retina_ai.apps.oct_screening` is the main OCT screening app with OTP login and clinical reasoning.
- `retina_ai.apps.secure_dr_oct` is a secure app with MFA login for combined DR and OCT inference.
- `retina_ai.scripts.train_oct8` trains the 8-class OCT model and saves weights, class mapping, and a confusion matrix.

## Dependencies

See `requirements.txt` for the project dependency list.

## Improvements made

- Converted the flat script layout into a package structure under `retina_ai/`
- Added `pyproject.toml` for installable package metadata
- Added `retina_ai.apps` and `retina_ai.scripts` entry points
- Updated dependency list and documentation

This repository is intended for research and demo purposes. Always validate machine learning predictions with qualified medical professionals.
