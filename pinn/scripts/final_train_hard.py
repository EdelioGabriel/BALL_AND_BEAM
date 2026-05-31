"""
Treino final da HardPINN com os melhores hiperparâmetros encontrados pelo Optuna
---------------------------------------------------------------------------------
Treina o modelo completo e salva os pesos para exportação ao firmware.

Uso:
    python final_train_hard.py
"""

import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import optuna

sys.path.append(str(Path(__file__).parent.parent))

from scripts.skeleton import HardPINN

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

print(f'Usando: {DEVICE}')

# ========================== Limites do domínio ======================

X_MIN    = 0.0
X_MAX    = 30.0
XDOT_MAX = 10.0
EINT_MAX = 50.0

# ========================== Amostragem no domínio ===================

def sample_domain(batch_size, device=DEVICE):
    x     = torch.FloatTensor(batch_size).uniform_(X_MIN,     X_MAX   ).to(device)
    x_dot = torch.FloatTensor(batch_size).uniform_(-XDOT_MAX, XDOT_MAX).to(device)
    x_ref = torch.FloatTensor(batch_size).uniform_(X_MIN,     X_MAX   ).to(device)
    e_int = torch.FloatTensor(batch_size).uniform_(-EINT_MAX, EINT_MAX).to(device)
    return torch.stack([x, x_dot, x_ref, e_int], dim=1)

# ========================== Função de loss ==========================

def loss_function(x_ddot_pred, x_ref, x, w_track=1.0, w_effort=0.1):
    erro        = (x_ref - x).unsqueeze(1)
    loss_track  = torch.mean((x_ddot_pred - erro) ** 2)
    loss_effort = torch.mean(x_ddot_pred ** 2)
    loss        = w_track * loss_track + w_effort * loss_effort
    return loss, loss_track, loss_effort

# ========================== Loop de treino ==========================

def train(model, optimizer, n_epochs, batch_size, w_track, w_effort):
    history = {'loss': [], 'loss_track': [], 'loss_effort': []}

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        X           = sample_domain(batch_size)
        theta       = model(X)
        x_ddot_pred = model.physics(theta)

        x     = X[:, 0]
        x_ref = X[:, 2]

        loss, loss_track, loss_effort = loss_function(
            x_ddot_pred, x_ref, x, w_track, w_effort
        )

        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['loss_track'].append(loss_track.item())
        history['loss_effort'].append(loss_effort.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch:05d} | Loss: {loss.item():.2e} | '
                  f'Track: {loss_track.item():.2e} | '
                  f'Effort: {loss_effort.item():.2e}')

    return history

# ====================================================================
# MELHORES HIPERPARÂMETROS — resultado do Optuna
# ====================================================================

OPTUNA_DIR = Path(__file__).parent / 'optuna'

study = optuna.load_study(
    study_name = 'hard_pinn_ball_and_beam',
    storage    = f'sqlite:///{OPTUNA_DIR}/hard_pinn_study.db'
)

BEST_PARAMS = study.best_params
print(f"Loss: {study.best_value:.4e}")
print(f"Params: {BEST_PARAMS}")

'''
BEST_PARAMS = {
    'batch_size': 256,
    'n_layers':   3,
    'n_hidden':   128,
    'lr':         0.001,
    'w_track':    8.0,
    'w_effort':   0.2,
    'activation': 'SiLU',
}
'''

ACTIVATION_MAP = {
    'Tanh': nn.Tanh,
    'SiLU': nn.SiLU,
}

# ====================================================================
# TREINO
# ====================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("Treino final HardPINN com melhores hiperparâmetros")
    print("=" * 50)
    for k, v in BEST_PARAMS.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    model = HardPINN(
        n_inputs   = 4,
        n_hidden   = BEST_PARAMS['n_hidden'],
        n_layers   = BEST_PARAMS['n_layers'],
        activation = ACTIVATION_MAP[BEST_PARAMS['activation']]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_PARAMS['lr'])

    history = train(
        model,
        optimizer,
        n_epochs   = 5000,
        batch_size = BEST_PARAMS['batch_size'],
        w_track    = BEST_PARAMS['w_track'],
        w_effort   = BEST_PARAMS['w_effort'],
    )

    # ================================================================
    # SALVA MODELO
    # ================================================================

    model_path = RESULTS_DIR / 'hard_pinn.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_inputs':   4,
            'n_hidden':   BEST_PARAMS['n_hidden'],
            'n_layers':   BEST_PARAMS['n_layers'],
            'activation': BEST_PARAMS['activation'],
        },
        'hyperparams': BEST_PARAMS,
        'final_loss':  history['loss'][-1],
    }, model_path)
    print(f"\nModelo salvo em: {model_path}")

    # ================================================================
    # GRÁFICOS
    # ================================================================

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(history['loss'],         label='total')
    ax.plot(history['loss_track'],   label='rastreamento', linestyle='--')
    ax.plot(history['loss_effort'],  label='esforço',      linestyle='-.')
    ax.set_yscale('log')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Histórico de loss — HardPINN')
    ax.legend()

    plt.tight_layout()
    plot_path = RESULTS_DIR / 'treino_final_hard_pinn.png'
    plt.savefig(plot_path)
    plt.show()
    print(f"Gráfico salvo em: {plot_path}")