"""
Treino da HardPINN para controle do sistema Ball and Beam.

Sem dados supervisionados e sem simulação. Pontos são amostrados
aleatoriamente no domínio [x, ẋ, x_ref, e_int] e a loss minimiza
o resíduo da EDP: ẍ - (5/9)·g·θ = 0.

Como a física está embutida na arquitetura via model.physics(theta),
o resíduo é satisfeito por construção — a loss guia os pesos para que
θ produza o comportamento de controle desejado.
"""

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================== Limites do domínio ======================

X_MIN    = 0.0      # posição mínima (cm)
X_MAX    = 30.0     # posição máxima (cm)
XDOT_MAX = 10.0     # velocidade máxima (cm/s)
EINT_MAX = 50.0     # limite do erro integrado

# ========================== Amostragem no domínio ===================

def sample_domain(batch_size, device=DEVICE):
    """
    Amostra pontos aleatórios no domínio de operação.

    Retorna X shape (batch_size, 4): [x, ẋ, x_ref, e_int]
    """
    x     = torch.FloatTensor(batch_size).uniform_(X_MIN,    X_MAX   ).to(device)
    x_dot = torch.FloatTensor(batch_size).uniform_(-XDOT_MAX, XDOT_MAX).to(device)
    x_ref = torch.FloatTensor(batch_size).uniform_(X_MIN,    X_MAX   ).to(device)
    e_int = torch.FloatTensor(batch_size).uniform_(-EINT_MAX, EINT_MAX).to(device)

    return torch.stack([x, x_dot, x_ref, e_int], dim=1)


# ========================== Função de loss ==========================

def loss_function(x_ddot_pred, x_ref, x, w_track=1.0, w_effort=0.1):
    """
    Loss baseada no resíduo do domínio — sem targets supervisionados.

    Args:
        x_ddot_pred: ẍ calculado via model.physics(theta)   shape (N, 1)
        x_ref:       setpoint amostrado                      shape (N,)
        x:           posição amostrada                       shape (N,)
        w_track:     peso do termo de rastreamento
        w_effort:    peso do esforço de controle

    Returns:
        loss, loss_track, loss_effort
    """
    # A aceleração deve apontar para reduzir o erro (x_ref - x)
    # ẍ_desejado ∝ (x_ref - x): se x < x_ref, ẍ deve ser positivo
    erro = (x_ref - x).unsqueeze(1)
    loss_track  = torch.mean((x_ddot_pred - erro) ** 2)

    # Penaliza esforço de controle desnecessário
    loss_effort = torch.mean(x_ddot_pred ** 2)

    loss = w_track * loss_track + w_effort * loss_effort

    return loss, loss_track, loss_effort


# ========================== Loop de treino ==========================

def train(model, optimizer, n_epochs, batch_size=512, w_track=1.0, w_effort=0.1):
    """
    Treina a HardPINN por amostragem no domínio.

    A cada época, novos pontos são sorteados — sem dataset fixo.
    A física ẍ = (5/9)·g·θ é satisfeita por construção via model.physics().

    Args:
        model:      rede HardPINN
        optimizer:  otimizador PyTorch
        n_epochs:   número de épocas
        batch_size: pontos amostrados por época
        w_track:    peso do rastreamento
        w_effort:   peso do esforço de controle
    """
    history = {
        'loss':        [],
        'loss_track':  [],
        'loss_effort': [],
    }

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Amostra pontos no domínio
        X = sample_domain(batch_size)

        # Forward: rede prediz θ
        theta = model(X)                        # shape (N, 1)

        # Hard constraint: ẍ = (5/9)·g·θ — nunca violado
        x_ddot_pred = model.physics(theta)      # shape (N, 1)

        # x e x_ref extraídos da entrada amostrada
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