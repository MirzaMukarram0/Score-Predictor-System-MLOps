# tests/test_train.py
import pandas as pd

from app.train import train_model


def test_train_model_on_synthetic():
    df = pd.DataFrame(
        {
            "Hours_Studied": [1, 2, 3, 4, 5, 6],
            "Attendance": [60, 65, 70, 75, 80, 85],
            "Previous_Scores": [50, 55, 60, 65, 70, 75],
            "Motivation_Level": [0, 1, 1, 2, 2, 2],
            "Exam_Score": [52, 58, 61, 68, 72, 78],
        }
    )
    model, metrics = train_model(df, test_size=0.33, random_state=1)
    assert "train_r2" in metrics

    preds = model.predict(
        df[
            [
                "Hours_Studied",
                "Attendance",
                "Previous_Scores",
                "Motivation_Level",
            ]
        ]
    )
    assert len(preds) == len(df)
