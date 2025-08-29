import pandas as pd
import numpy as np

n_samples = 5000
np.random.seed(42)

data = {
        'provinsi_id': np.random.randint(1, 35, n_samples),  # 34 provinsi
        'kabupaten_id': np.random.randint(1, 515, n_samples),  # ~514 kab/kota
        'kepadatan_penduduk': np.random.lognormal(5, 1.5, n_samples),
        'tingkat_pendidikan': np.random.normal(8.5, 2.5, n_samples),  # rata2 tahun sekolah
        'tingkat_pengangguran': np.random.gamma(2, 3, n_samples),  # %
        'akses_air_bersih': np.random.beta(7, 2, n_samples) * 100,  # %
        'akses_listrik': np.random.beta(8, 1.5, n_samples) * 100,  # %
        'fasilitas_kesehatan': np.random.poisson(15, n_samples),  # per 10k penduduk
        'jalan_aspal': np.random.beta(5, 3, n_samples) * 100,  # %
        'luas_sawah': np.random.exponential(5000, n_samples),  # hektar
        'pendapatan_perkapita': np.random.lognormal(14.5, 0.8, n_samples),  # rupiah/bulan
        'jenis_wilayah': np.random.choice(['urban', 'rural'], n_samples, p=[0.6, 0.4])
    }

df = pd.DataFrame(data)

print(df.head())

'''
kemiskinan = 1 jika pendapatan_perkapita < Rp1.800.000 dan
- minimal dua dari tiga kondisi berikut terpenuhi:
➔ akses_air_bersih < 80%
➔ akses_listrik < 85%
➔ tingkat_pendidikan < 8 tahun
- kemiskinan = 0 untuk kasus lainnya.
'''

def label_kemiskinan(row):
    kondisi_A = row['pendapatan_perkapita'] < 1800000
    kondisi_B = [
        row['akses_air_bersih'] < 80,
        row['akses_listrik'] < 85,
        row['tingkat_pendidikan'] < 8
    ]
    if kondisi_A and sum(kondisi_B) >= 2:
        return 1
    return 0

df["kemiskinan"] = df.apply(label_kemiskinan, axis=1)
df.to_csv('dataset.csv', index=False)
print("data berhasil dibuat")
    
    