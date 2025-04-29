import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


heartDisease = pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv")

def discretize(data, column, bins, labels):
    data[column] = pd.cut(data[column], bins=bins, labels=labels)
    return data

heartDisease = discretize(heartDisease, 'age', [0, 40, 60, 100], ['young', 'middle', 'old'])
heartDisease = discretize(heartDisease, 'chol', [0, 200, 240, 260], ['normal', 'borderline', 'high'])
heartDisease = discretize(heartDisease, 'thalach', [0, 120, 150, 220], ['low', 'normal', 'high'])


columns = ["age", "chol", "fbs", "restecg", "thalach", "target"]
heartDisease = heartDisease[columns]
for col in columns:
    heartDisease[col] = heartDisease[col].astype("category")


model = DiscreteBayesianNetwork([
    ("age", "fbs"),
    ("fbs", "target"),
    ("target", "restecg"),
    ("target", "thalach"),
    ("target", "chol")
])
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(model)
q = infer.query(variables=["target"], evidence={"age": "young"})
print(q)
