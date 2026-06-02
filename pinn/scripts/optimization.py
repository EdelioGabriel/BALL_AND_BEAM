'''
Este script tem como objetivo realizar a otimização dos hiperparâmetros da rede através do framework do Optuna

- Nota do autor
'''

import sys
import optuna
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.train import simulate, loss_function, sample_initial_conditions
from scripts.skeleton import LikePINN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando: {DEVICE}')

# ====================================================================
# OBJETIVO
# ====================================================================

def objective(trial):

    # ── Hiperparâmetros a otimizar ───────────────────────────────────
    n_epochs   = 500
    batch_size = trial.suggest_int('batch_size', 100, 1000)
    n_steps    = trial.suggest_int('n_steps',    60, 200)
    n_layers   = trial.suggest_int('n_layers',   1, 3)
    n_hidden   = trial.suggest_categorical('n_hidden', [64, 128, 256])
    lr         = trial.suggest_float('lr',       1e-4, 1e-2, log=True)
    w_edo      = trial.suggest_float('w_edo',    1, 5.0)
    w_state    = trial.suggest_float('w_state',  5, 10.0)
    w_effort   = trial.suggest_float('w_effort', 0.1, 0.5)
    activation_name = trial.suggest_categorical('activation', ['Tanh', 'SiLU'])

    activation_map = {
        'Tanh': nn.Tanh,
        'SiLU': nn.SiLU,
    }

    # ── Modelo e otimizador ──────────────────────────────────────────
    model = LikePINN(
        n_inputs   = 4,
        n_outputs  = 1,
        n_hidden   = n_hidden,
        n_layers   = n_layers,
        activation = activation_map[activation_name]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Loop de treino com reporte ao pruner ─────────────────────────
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        x0, x_dot0, x_ref = sample_initial_conditions(batch_size, device=DEVICE)
        xs, thetas, _      = simulate(model, x0, x_dot0, x_ref, n_steps, device=DEVICE)

        loss, _, _, _ = loss_function(
            xs, thetas, x_ref,
            w_state  = w_state,
            w_effort = w_effort,
            w_edo    = w_edo,
        )

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
        n_startup_trials = 20,   # mediana mais estável antes de podar
        n_warmup_steps   = 100,  # alinhado com n_epochs mínimo de 300
        interval_steps   = 5,   # avalia a cada 10 épocas
    )
    sampler = optuna.samplers.TPESampler(seed=367)

    OPTUNA_DIR = Path(__file__).parent / 'optuna'
    OPTUNA_DIR.mkdir(exist_ok=True)

    study = optuna.create_study(
        direction      = 'minimize',
        sampler        = sampler,
        pruner         = pruner,
        study_name     = 'likepinn_ball_and_beam',
        storage        = f'sqlite:///{OPTUNA_DIR}/likepinn_study.db',
        load_if_exists = True
    )

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\n" + "=" * 50)
    print("Melhor trial:")
    print(f"  Loss:   {study.best_value:.4e}")
    print(f"  Params: {study.best_params}")
    print("=" * 50)