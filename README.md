# Haggle Hour Detective

Predicts the number of hours spent on a project using machine learning.

## Model
- CatBoost Regressor
- Gradient Boosting Decision Trees
- Target to predict: `hours_spent`

## Usage

Install dependecies:

```bash
pip install -r requirements.txt
```
Train the model:

```bash
python3 train.py
```

Generate predictions:

```bash
python3 predict.py
```

Predictions are saved to `predictions.csv`.

## Libraries

- pandas
- numpy
- catboost
- scikit-learn
