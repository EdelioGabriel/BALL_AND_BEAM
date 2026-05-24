"""
Script de validação do treino da PINN - Ball and Beam
------------------------------------------------------
Treinamento por simulação diferenciável.

Verifica se o pipeline de treino está funcionando corretamente:
- Loss total decrescente
- Loss de rastreamento decrescente
- Loss física decrescente
- Trajetória simulada convergindo para x_ref

Uso:
    python test_train.py
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

# Adiciona pinn/ ao path para importar os módulos
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train import train, simulate, sample_initial_conditions, DT
from scripts.skeleton import PINN

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ========================== Configuração do teste ==========================

N_EPOCHS_TEST = 1000
BATCH_SIZE    = 64
N_STEPS       = 40      # 40 × 0.05s = 2 segundos por episódio
LR            = 1e-3
W_STATE       = 1.0
W_EFFORT      = 0.1
W_PDE         = 0.1

# ========================== Execução =======================================

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Épocas: {N_EPOCHS_TEST} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"n_steps: {N_STEPS} ({N_STEPS * DT:.1f}s por episódio)")
    print(f"w_state: {W_STATE} | w_effort: {W_EFFORT} | w_pde: {W_PDE}")
    print("=" * 50)

    model = PINN(
        n_inputs=4,
        n_outputs=1,
        n_hidden=32,
        n_layers=2,
        activation=nn.Tanh
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = train(
        model,
        optimizer,
        n_epochs   = N_EPOCHS_TEST,
        batch_size = BATCH_SIZE,
        n_steps    = N_STEPS,
        w_state    = W_STATE,
        w_effort   = W_EFFORT,
        w_pde      = W_PDE,
    )

    # ========================== Verificações ================================

    print("\n" + "=" * 50)
    print("Verificações:")

    loss_caiu  = history['loss'][-1]        < history['loss'][0]
    state_caiu = history['loss_state'][-1]  < history['loss_state'][0]
    pde_caiu   = history['loss_pde'][-1]    < history['loss_pde'][0]

    print(f"  Loss total decrescente:       {'✓' if loss_caiu else '✗'} "
          f"({history['loss'][0]:.4e} → {history['loss'][-1]:.4e})")
    print(f"  Loss rastreamento decrescente:{'✓' if state_caiu else '✗'} "
          f"({history['loss_state'][0]:.4e} → {history['loss_state'][-1]:.4e})")
    print(f"  Loss física decrescente:      {'✓' if pde_caiu else '✗'} "
          f"({history['loss_pde'][0]:.4e} → {history['loss_pde'][-1]:.4e})")

    # Verifica trajetória simulada em um exemplo fixo
    model.eval()
    with torch.no_grad():
        x0     = torch.tensor([2.0],  device=DEVICE)
        x_dot0 = torch.tensor([0.0],  device=DEVICE)
        x_ref  = torch.tensor([15.0], device=DEVICE)

        xs, thetas, _ = simulate(model, x0, x_dot0, x_ref, n_steps=80)

        x_final = xs[0, -1].item()
        erro_final = abs(x_final - x_ref.item())
        convergiu = erro_final < 2.0   # tolerância de 2 cm

        print(f"  Trajetória x0=2cm → x_ref=15cm:")
        print(f"    x_final={x_final:.2f}cm | erro={erro_final:.2f}cm | "
              f"{'✓ convergiu' if convergiu else '✗ não convergiu'}")

    # ========================== Gráficos ====================================

    t_sim = [i * DT for i in range(xs.shape[1])]
    t_ctrl = [i * DT for i in range(thetas.shape[1])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histórico de loss
    axes[0].plot(history['loss'],        label='total')
    axes[0].plot(history['loss_state'],  label='rastreamento', linestyle='--')
    axes[0].plot(history['loss_effort'], label='esforço',      linestyle='-.')
    axes[0].plot(history['loss_pde'],    label='física',       linestyle=':')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Histórico de loss')
    axes[0].legend()

    # Trajetória simulada
    axes[1].plot(t_sim, xs[0].cpu().numpy(), label='x(t)')
    axes[1].axhline(x_ref.item(), color='r', linestyle='--', label='x_ref')
    axes[1].set_xlabel('Tempo (s)')
    axes[1].set_ylabel('Posição (cm)')
    axes[1].set_title('Trajetória simulada (x0=2cm, x_ref=15cm)')
    axes[1].legend()

    # Ação de controle
    axes[2].plot(t_ctrl, thetas[0].cpu().numpy(), label='θ(t)')
    axes[2].axhline(0, color='k', linestyle=':', alpha=0.3)
    axes[2].set_xlabel('Tempo (s)')
    axes[2].set_ylabel('Ângulo (graus)')
    axes[2].set_title('Sinal de controle')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test_train.png')
    plt.show()
    print(f"\nGráfico salvo em: {RESULTS_DIR / 'test_train.png'}")

# Salva o modelo
model_path = RESULTS_DIR / 'pinn_ball_beam.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'n_inputs': 4,
        'n_outputs': 1,
        'n_hidden': 32,
        'n_layers': 2,
    }
}, model_path)
print(f"Modelo salvo em: {model_path}")