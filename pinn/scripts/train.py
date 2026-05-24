"""
Este Script contém o código para realizar o treino da Rede Neural Informada por Física (PINN)
que tem como objetivo aprender uma política de controle para o sistema Ball and Beam
através de treinamento em simulação.

Treinamento puramente baseado em simulação diferenciável.
"""

import torch
import numpy as np
from pathlib import Path

# Definição do local onde o código será executado. Por padrão, gpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================== Constantes físicas ======================

G        = 9.8      # gravidade (m/s²)
ALPHA    = 5/9      # coeficiente da equação linearizada
DT       = 0.05     # período de amostragem (s) - o mesmo do controle implementado (50 ms)
X_MIN    = 0.0      # posição mínima (cm)
X_MAX    = 30.0     # posição máxima (cm)
XDOT_MAX = 10.0     # velocidade máxima (cm/s)
THETA_MAX = 60.0    # ângulo máximo comandado (graus)

# ========================== Simulador diferenciável =================

def simulate(model, x0, x_dot0, x_ref, n_steps, device=DEVICE):
    """
    Simula o sistema ball and beam com a política da rede.

    Args:
        model:   rede PINN
        x0:      posição inicial        shape (N,)
        x_dot0:  velocidade inicial     shape (N,)
        x_ref:   setpoint               shape (N,)
        n_steps: número de passos de simulação

    Returns:
        xs:      trajetória de posições  shape (N, n_steps+1)
        thetas:  ações aplicadas         shape (N, n_steps)
        e_ints:  erro integrado          shape (N, n_steps)
    """
    N = x0.shape[0]

    x     = x0.clone()
    x_dot = x_dot0.clone()
    e_int = torch.zeros(N, device=device)

    xs     = [x.unsqueeze(1)]
    thetas = []
    e_ints = []

    for _ in range(n_steps):
        # Atualiza erro integrado
        e_int = e_int + (x_ref - x) * DT

        # Monta entrada da rede: [x, ẋ, x_ref, e_int]
        X_input = torch.stack([x, x_dot, x_ref, e_int], dim=1)

        # Rede produz θ (em graus normalizados)
        theta = model(X_input).squeeze(1)

        # Limita ângulo ao range físico
        theta = torch.clamp(theta, -THETA_MAX, THETA_MAX)

        # Adiciona a equação do movimento: ẍ = (5/9)·g·θ
        x_ddot = ALPHA * G * theta
        x_dot  = x_dot + x_ddot * DT
        x      = x + x_dot * DT

        # Limita posição ao range físico
        x = torch.clamp(x, X_MIN, X_MAX)

        xs.append(x.unsqueeze(1))
        thetas.append(theta.unsqueeze(1))
        e_ints.append(e_int.unsqueeze(1))

    xs     = torch.cat(xs,     dim=1)   # (N, n_steps+1)
    thetas = torch.cat(thetas, dim=1)   # (N, n_steps)
    e_ints = torch.cat(e_ints, dim=1)   # (N, n_steps)

    return xs, thetas, e_ints

# ========================== Função de loss ==========================

def loss_function(xs, thetas, x_ref, w_state=1.0, w_effort=0.1, w_pde=0.1):
    """
    xs:      trajetória simulada    shape (N, n_steps+1)
    thetas:  ações aplicadas        shape (N, n_steps)
    x_ref:   setpoint               shape (N,)
    """
    # Loss de rastreamento — x deve convergir para x_ref
    x_ref_expanded = x_ref.unsqueeze(1).expand_as(xs)
    loss_state = torch.mean((xs - x_ref_expanded) ** 2)

    # Loss de esforço — penaliza ângulos grandes
    loss_effort = torch.mean(thetas ** 2)

    # Resíduo físico — verifica consistência da trajetória com a EDP
    # ẍ estimado da trajetória simulada
    x_ddot_sim = torch.diff(torch.diff(xs, dim=1), dim=1) / (DT ** 2)
    theta_mid  = thetas[:, :-1]
    residual   = x_ddot_sim - ALPHA * G * theta_mid
    loss_pde   = torch.mean(residual ** 2)

    loss = w_state * loss_state + w_effort * loss_effort + w_pde * loss_pde

    return loss, loss_state, loss_effort, loss_pde

# ========================== Amostragem de condições iniciais ========

def sample_initial_conditions(batch_size, device=DEVICE):
    """
    Sorteia condições iniciais aleatórias dentro do espaço de operação.
    """
    x0     = torch.FloatTensor(batch_size).uniform_(X_MIN, X_MAX).to(device)
    x_dot0 = torch.FloatTensor(batch_size).uniform_(-XDOT_MAX, XDOT_MAX).to(device)
    x_ref  = torch.FloatTensor(batch_size).uniform_(X_MIN, X_MAX).to(device)

    return x0, x_dot0, x_ref

# ========================== Loop de treino ==========================

def train(model, optimizer, n_epochs, batch_size=64, n_steps=40,
          w_state=1.0, w_effort=0.1, w_pde=0.1):
    """
    Treina a PINN por simulação diferenciável.

    Args:
        model:      rede PINN
        optimizer:  otimizador PyTorch
        n_epochs:   número de épocas
        batch_size: condições iniciais por batch
        n_steps:    passos de simulação por episódio (n_steps * DT = duração em segundos)
        w_state:    peso do rastreamento
        w_effort:   peso do esforço de controle
        w_pde:      peso do resíduo físico
    """
    history = {
        'loss':        [],
        'loss_state':  [],
        'loss_effort': [],
        'loss_pde':    [],
    }

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Sorteia condições iniciais
        x0, x_dot0, x_ref = sample_initial_conditions(batch_size)

        # Simula com a política atual
        xs, thetas, _ = simulate(model, x0, x_dot0, x_ref, n_steps)

        # Calcula loss
        loss, loss_state, loss_effort, loss_pde = loss_function(
            xs, thetas, x_ref, w_state, w_effort, w_pde
        )

        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['loss_state'].append(loss_state.item())
        history['loss_effort'].append(loss_effort.item())
        history['loss_pde'].append(loss_pde.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch:05d} | Loss: {loss.item():.2e} | '
                  f'state: {loss_state.item():.2e} | '
                  f'Effort: {loss_effort.item():.2e} | '
                  f'PDE: {loss_pde.item():.2e}')

    return history