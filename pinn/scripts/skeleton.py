"""
Este script contém todas as funções e estruturas gerais para a implementação dos exemplos,
evitando repetição desnecessária.

Arquitetura inspirada em:
    Raissi et al. (2017) — https://arxiv.org/abs/1711.10561
    Baty (2024)          — https://arxiv.org/abs/2403.00599

- Nota do autor
"""

import torch
import torch.nn as nn

# ====================================================================
# CONSTANTES FÍSICAS
# ====================================================================
G     = 9.8    # gravidade (m/s²)
ALPHA = 5/9    # coeficiente da equação linearizada Ball and Beam

# ====================================================================
# ARQUITETURA BASE — MLP
# ====================================================================

class LikePINN(nn.Module):
    """
    Arquitetura MLP inspirada em Physics-Informed Neural Networks.
    A física do sistema é incorporada no processo de treinamento,
    como termo de penalização na loss.

    Recebe como parâmetros de arquitetura:
        n_inputs:   Quantidade de dados de entrada
        n_outputs:  Quantidade de dados de saída
        n_hidden:   Quantidade de neurônios nas camadas ocultas
        n_layers:   Quantidade de camadas
        activation: Função de ativação

    Retorna: O vetor X após o passo forward
    """
    def __init__(self, n_inputs, n_outputs, n_hidden, n_layers, activation):
        super().__init__()

        layers = []
        layers.append(nn.Linear(n_inputs, n_hidden))
        layers.append(activation())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activation())

        layers.append(nn.Linear(n_hidden, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)