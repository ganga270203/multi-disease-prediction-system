import pandas as pd
import numpy as np

np.random.seed(42)
n = 400

df = pd.DataFrame({
    'id': range(1, n + 1),
    'age': np.random.randint(20, 80, n),
    'bp': np.random.choice([60, 70, 80, 90, 100], n),
    'sg': np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n),
    'al': np.random.choice([0, 1, 2, 3, 4], n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
    'su': np.random.choice([0, 1, 2, 3, 4], n, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
    'bgr': np.random.randint(70, 490, n),
    'bu': np.random.randint(10, 200, n),
    'sc': np.round(np.random.uniform(0.4, 15.0, n), 1),
    'sod': np.random.randint(110, 160, n),
    'pot': np.round(np.random.uniform(2.5, 7.5, n), 1),
    'hemo': np.round(np.random.uniform(3.1, 17.8, n), 1),
    'pcv': np.random.randint(15, 54, n),
    'wbcc': np.random.randint(2200, 26400, n),
    'rbcc': np.round(np.random.uniform(2.1, 6.5, n), 1),
    'classification': np.random.choice([0, 1], n, p=[0.38, 0.62])
})

df.to_csv('data/raw/kidney.csv', index=False)
print("Successfully created data/raw/kidney.csv with 400 clinical records!")