import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('dataset.csv')

df.info()

X = df.drop('kemiskinan',axis=1)
y = df['kemiskinan']

numeric_features = ["kepadatan_penduduk","tingkat_pendidikan","tingkat_pengangguran","akses_air_bersih","akses_listrik","fasilitas_kesehatan","jalan_aspal","luas_sawah","pendapatan_perkapita"]
categorical_features = ['jenis_wilayah']

preprocessor =  ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features ),
           ],
    remainder='drop'
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

X_processed = pipeline.fit_transform(X)

joblib.dump(pipeline, 'preprocessor.pkl')
print("Preprocessor berhasil dibuat")
