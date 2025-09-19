# src/train.py
import os
import argparse
import pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# FEATURES expected in the CSV
FEATURES = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Motivation_Level']
TARGET = 'Exam_Score'


def load_data(csv_path):
    """Load CSV and perform minimal preprocessing expected by the model."""
    df = pd.read_csv(csv_path)
    # map Motivation_Level to numeric if it's categorical strings
    if df.get('Motivation_Level') is not None:
        df['Motivation_Level'] = df['Motivation_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    return df


def train_model(df, test_size=0.2, random_state=42):
    """Train LinearRegression and return (model, metrics_dict)."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "train_mae": round(mean_absolute_error(y_train, train_pred), 2),
        "train_mse": round(mean_squared_error(y_train, train_pred), 2),
        "train_rmse": round(np.sqrt(mean_squared_error(y_train, train_pred)), 2),
        "train_r2": round(r2_score(y_train, train_pred), 4),
        "test_mae": round(mean_absolute_error(y_test, test_pred), 2),
        "test_mse": round(mean_squared_error(y_test, test_pred), 2),
        "test_rmse": round(np.sqrt(mean_squared_error(y_test, test_pred)), 2),
        "test_r2": round(r2_score(y_test, test_pred), 4),
    }

    return model, metrics


def save_model(model, out_path):
    """Save model as pickle to out_path (create folder if needed)."""
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)


def predict_example(model):
    example = pd.DataFrame([{
        "Hours_Studied": 8,
        "Attendance": 90,
        "Previous_Scores": 75,
        "Motivation_Level": 2
    }])
    pred = model.predict(example)
    return float(pred[0])


def main():
    parser = argparse.ArgumentParser(description="Train linear regression for Student Performance")
    parser.add_argument("--data", default="../datasets/StudentPerformanceFactors.csv",
                        help="Path to CSV dataset (default: ../datasets/StudentPerformanceFactors.csv)")
    parser.add_argument("--out", default="../datasets/student_model.pkl",
                        help="Path to save trained model pickle (default: ../datasets/student_model.pkl)")
    args = parser.parse_args()

    df = load_data(args.data)
    model, metrics = train_model(df)

    save_model(model, args.out)

    # Print metrics
    print("For Train :")
    print("MAE:", metrics["train_mae"])
    print("MSE:", metrics["train_mse"])
    print("RMSE:", metrics["train_rmse"])
    print("R-squared:", metrics["train_r2"])

    print("\nFor Test:")
    print("MAE:", metrics["test_mae"])
    print("MSE:", metrics["test_mse"])
    print("RMSE:", metrics["test_rmse"])
    print("R-squared:", metrics["test_r2"])

    # Example prediction
    print(f"\nPredicted Exam Score for sample example: {predict_example(model)}")


if __name__ == "__main__":
    main()
