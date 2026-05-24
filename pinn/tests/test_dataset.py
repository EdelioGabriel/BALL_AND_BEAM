import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'processed_data'

# ========================== Funções auxiliares ======================


csv_files = sorted(DATA_DIR.glob('dataset_processado*.csv'))
df = pd.read_csv(csv_files[-1])

import matplotlib.pyplot as plt
df['u_K_norm'].hist(bins=50)
plt.xlabel('u_K_norm')
plt.title('Distribuição do sinal de controle')
plt.show()