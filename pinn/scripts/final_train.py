"""
Treino final da PINN com os melhores hiperparâmetros encontrados pelo Optuna
----------------------------------------------------------------------------
Treina o modelo completo e salva os pesos para exportação ao firmware.

Uso:
    python treino_final.py
"""

import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import optuna

sys.path.append(str(Path(__file__).parent.parent))

from scripts.train import train, simulate, DT
from scripts.skeleton import LikePINN

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

print(f'Usando: {DEVICE}')

# ====================================================================
# MELHORES HIPERPARÂMETROS — resultado do Optuna
# ====================================================================

OPTUNA_DIR = Path(__file__).parent / 'optuna'

study = optuna.load_study(
    study_name = 'likepinn_ball_and_beam',
    storage    = f'sqlite:///{OPTUNA_DIR}/likepinn_study.db'
)

BEST_PARAMS = study.best_params
print(f"Loss: {study.best_value:.4e}")
print(f"Params: {BEST_PARAMS}")
'''

BEST_PARAMS = {
    'n_epochs':   474,
    'batch_size': 128,
    'n_steps':    84,
    'n_layers':   2,
    'n_hidden':   128,
    'lr':         0.001279411173119678,
    'w_edo':      0.0,
    'w_state':    10.0,
    'w_effort':   0.5,
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
    print("Treino final com melhores hiperparâmetros")
    print("=" * 50)
    for k, v in BEST_PARAMS.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    model = LikePINN(
        n_inputs   = 4,
        n_outputs  = 1,
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
        n_steps    = BEST_PARAMS['n_steps'],
        w_state    = BEST_PARAMS['w_state'],
        w_effort   = BEST_PARAMS['w_effort'],
        w_edo      = BEST_PARAMS['w_edo'],
    )

    # ================================================================
    # SALVA MODELO
    # ================================================================

    model_path = RESULTS_DIR / 'likepinn_best.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_inputs':   4,
            'n_outputs':  1,
            'n_hidden':   BEST_PARAMS['n_hidden'],
            'n_layers':   BEST_PARAMS['n_layers'],
            'activation': BEST_PARAMS['activation'],
        },
        'hyperparams':  BEST_PARAMS,
        'final_loss':   history['loss'][-1],
    }, model_path)
    print(f"\nModelo salvo em: {model_path}")

    # ================================================================
    # VALIDAÇÃO — trajetória simulada
    # ================================================================

    model.eval()
    with torch.no_grad():
        x0     = torch.tensor([2.0],  device=DEVICE)
        x_dot0 = torch.tensor([0.0],  device=DEVICE)
        x_ref  = torch.tensor([15.0], device=DEVICE)
        xs, thetas, _ = simulate(model, x0, x_dot0, x_ref, n_steps=120)

    t_sim  = [i * DT for i in range(xs.shape[1])]
    t_ctrl = [i * DT for i in range(thetas.shape[1])]

    # ================================================================
    # GRÁFICOS
    # ================================================================

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['loss'],        label='total')
    axes[0].plot(history['loss_state'],  label='rastreamento', linestyle='--')
    axes[0].plot(history['loss_effort'], label='esforço',      linestyle='-.')
    axes[0].plot(history['loss_edo'],    label='física',       linestyle=':')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Histórico de loss')
    axes[0].legend()

    axes[1].plot(t_sim, xs[0].cpu().numpy(), label='x(t)')
    axes[1].axhline(x_ref.item(), color='r', linestyle='--', label='x_ref')
    axes[1].set_xlabel('Tempo (s)')
    axes[1].set_ylabel('Posição (cm)')
    axes[1].set_title('Trajetória simulada (x0=2cm, x_ref=15cm)')
    axes[1].legend()

    axes[2].plot(t_ctrl, thetas[0].cpu().numpy(), label='θ(t)')
    axes[2].axhline(0, color='k', linestyle=':', alpha=0.3)
    axes[2].set_xlabel('Tempo (s)')
    axes[2].set_ylabel('Ângulo (graus)')
    axes[2].set_title('Sinal de controle')
    axes[2].legend()

    plt.tight_layout()
    plot_path = RESULTS_DIR / 'treino_final_likepinn.png'
    plt.savefig(plot_path)
    plt.show()
    print(f"Gráfico salvo em: {plot_path}")