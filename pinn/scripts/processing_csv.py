'''
Este script tem como objetivo realizar o pré-processamento dos dados coletados 
pelo script "generate_data.py" e armazenados no arquivo .csv, na pasta "data"
'''

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from datetime import datetime

# Caminho relativo ao script
RAW_DATA_DIR = Path(__file__).parent.parent / 'raw_data'
PROCESSED_DATA_DIR = Path(__file__).parent.parent / 'processed_data'

# Carrega o CSV mais recente da pasta
csv_files = sorted(RAW_DATA_DIR.glob('coleta_*.csv'))
df = pd.read_csv(csv_files[-1])

print(f"Arquivo carregado: {csv_files[-1].name}")
print(f"Amostras: {len(df)}")
print(df.head())

DT = 0.05  # período de amostragem

# Filtra ẋ antes de derivar
velocidade_filtrada = savgol_filter(df['velocidade_cms'].values, window_length=21,  # janela em amostras
polyorder=3)

# Recalcula ẍ com a velocidade filtrada
df['x_ddot'] = np.gradient(velocidade_filtrada, DT)
x_ddot_mean = df['x_ddot'].mean()
x_ddot_std  = df['x_ddot'].std()
df['x_ddot_norm'] = (df['x_ddot'] - x_ddot_mean) / x_ddot_std

# Normalização (zero mean, unit std)
features = ['posicao_cm', 'velocidade_cms', 'x_ref_cm', 'e_int']
stats = {}

for col in features:
    mean = df[col].mean()
    std  = df[col].std()
    df[col + '_norm'] = (df[col] - mean) / std
    stats[col] = {'mean': mean, 'std': std}

# u_K normalizado separadamente (saída da rede)
stats['u_K'] = {'mean': df['u_K'].mean(), 'std': df['u_K'].std()}
df['u_K_norm'] = (df['u_K'] - stats['u_K']['mean']) / stats['u_K']['std']

print(df[['posicao_cm_norm', 'velocidade_cms_norm', 'x_ref_cm_norm', 'e_int_norm', 'u_K_norm', 'x_ddot']].describe())
print("\nEstatísticas de normalização:")
for k, v in stats.items():
    print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")


# Salva o dataset processado
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = PROCESSED_DATA_DIR / f'dataset_processado{timestamp}.csv'
df.to_csv(output_path, index=False)
print(f"\nDataset salvo em: {output_path}")
