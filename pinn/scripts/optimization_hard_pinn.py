'''
Otimização de hiperparâmetros da HardPINN para controle do sistema Ball and Beam.
    python optimization_hard.py
'''

import sys
import optuna
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.skeleton import HardPINN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando: {DEVICE}')

# ========================== Limites do domínio ======================

X_MIN    = 0.0
X_MAX    = 30.0
XDOT_MAX = 10.0
EINT_MAX = 50.0

# ========================== Amostragem no domínio ===================

def sample_domain(batch_size, device=DEVICE):
    """
    Amostra pontos aleatórios no domínio de operação.
    Retorna X shape (batch_size, 4): [x, ẋ, x_ref, e_int]
    """
    x     = torch.FloatTensor(batch_size).uniform_(X_MIN,     X_MAX   ).to(device)
    x_dot = torch.FloatTensor(batch_size).uniform_(-XDOT_MAX, XDOT_MAX).to(device)
    x_ref = torch.FloatTensor(batch_size).uniform_(X_MIN,     X_MAX   ).to(device)
    e_int = torch.FloatTensor(batch_size).uniform_(-EINT_MAX, EINT_MAX).to(device)

    return torch.stack([x, x_dot, x_ref, e_int], dim=1)

# ========================== Função de loss ==========================

def loss_function(x_ddot_pred, x_ref, x, w_track=1.0, w_effort=0.1):
    """
    Loss baseada em resíduo — sem targets supervisionados.

    A aceleração predita deve ser proporcional ao erro (x_ref - x).
    """
    erro        = (x_ref - x).unsqueeze(1)
    loss_track  = torch.mean((x_ddot_pred - erro) ** 2)
    loss_effort = torch.mean(x_ddot_pred ** 2)

    return w_track * loss_track + w_effort * loss_effort

# ====================================================================
# OBJETIVO
# ====================================================================

def objective(trial):

    # ── Hiperparâmetros a otimizar ───────────────────────────────────
    n_epochs   = 500
    batch_size = trial.suggest_int  ('batch_size', 100, 1000)
    n_layers   = trial.suggest_int  ('n_layers',   2, 5)
    n_hidden   = trial.suggest_categorical('n_hidden', [64, 128, 256, 512])
    lr         = trial.suggest_float('lr',       1e-4, 1e-2, log=True)
    w_track    = trial.suggest_float('w_track',  1.0, 10.0)
    w_effort   = trial.suggest_float('w_effort', 0.1, 0.5)
    activation_name = trial.suggest_categorical('activation', ['Tanh', 'SiLU'])

    activation_map = {
        'Tanh': nn.Tanh,
        'SiLU': nn.SiLU,
    }

    # ── Modelo e otimizador ──────────────────────────────────────────
    model = HardPINN(
        n_inputs   = 4,
        n_hidden   = n_hidden,
        n_layers   = n_layers,
        activation = activation_map[activation_name]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Loop de treino com reporte ao pruner ─────────────────────────
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        X = sample_domain(batch_size, device=DEVICE)

        # Hard constraint: física embutida na arquitetura
        theta       = model(X)
        x_ddot_pred = model.physics(theta)

        x     = X[:, 0]
        x_ref = X[:, 2]

        loss = loss_function(x_ddot_pred, x_ref, x, w_track, w_effort)

        loss.backward()
        optimizer.step()

        # Reporta para o pruner — interrompe se o trial for ruim
        trial.report(loss.item(), epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss.item()

# ====================================================================
# EXECUÇÃO
# ====================================================================
print("Iniciando otimização de hiperparâmetros...")
if __name__ == '__main__':
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials = 20,
        n_warmup_steps   = 100,
        interval_steps   = 5,
    )
    sampler = optuna.samplers.TPESampler(seed=367)

    OPTUNA_DIR = Path(__file__).parent / 'optuna'
    OPTUNA_DIR.mkdir(exist_ok=True)

    study = optuna.create_study(
        direction      = 'minimize',
        sampler        = sampler,
        pruner         = pruner,
        study_name     = 'hard_pinn_ball_and_beam_2',
        storage        = f'sqlite:///{OPTUNA_DIR}/hard_pinn_study.db',
        load_if_exists = True
    )

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\n" + "=" * 50)
    print("Melhor trial:")
    print(f"  Loss:   {study.best_value:.4e}")
    print(f"  Params: {study.best_params}")
    print("=" * 50)