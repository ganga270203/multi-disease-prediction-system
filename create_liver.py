import pandas as pd
import numpy as np

np.random.seed(42)
n = 583

df = pd.DataFrame({
    'Age': np.random.randint(10, 85, n),
    'Gender': np.random.choice([0, 1], n, p=[0.25, 0.75]),
    'Total_Bilirubin': np.round(np.random.uniform(0.4, 28.0, n), 2),
    'Direct_Bilirubin': np.round(np.random.uniform(0.1, 12.0, n), 2),
    'Alkaline_Phosphotase': np.random.randint(60, 1500, n),
    'Alamine_Aminotransferase': np.random.randint(10, 1000, n),
    'Aspartate_Aminotransferase': np.random.randint(10, 1200, n),
    'Total_Protiens': np.round(np.random.uniform(2.7, 9.6, n), 2),
    'Albumin': np.round(np.random.uniform(0.9, 5.5, n), 2),
    'Albumin_and_Globulin_Ratio': np.round(np.random.uniform(0.3, 2.8, n), 2),
    'Dataset': np.random.choice([0, 1], n, p=[0.29, 0.71])
})

df.to_csv('data/raw/liver.csv', index=False)
print("Successfully created data/raw/liver.csv with 583 patient records!")