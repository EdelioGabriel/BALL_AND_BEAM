"""
Este script contém todas as funções e estruturas gerais para a implementação dos exemplos, evitando repetição desnecessária

- Nota do autor
"""

import torch
import torch.nn as nn
import numpy as np

# ====================================================================
# CONSTANTES FÍSICAS
# ====================================================================
G     = 9.8    # gravidade
ALPHA = 5/9    # coeficiente da equação linearizada Ball and Beam


# ====================================================================
# ARQUITETURA BASE — MLP
# ====================================================================

class PINN(nn.Module):
    """
    Classe para definir a arquitetura MLP (Soft-PINN)
    A física entra como termo de penalização na loss — restrição suave.

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


# ====================================================================
# HARD-PINN — física embutida na arquitetura
# ====================================================================

class HardPINN(nn.Module):
    """
    Hard-PINN para controle do sistema Ball and Beam.

    A restrição física ẍ = (5/9)·g·θ é satisfeita por construção:
    a rede aprende θ e ẍ é calculado diretamente pela equação física,
    nunca podendo violar a lei do movimento.

    Recebe como parâmetros de arquitetura:
        n_inputs:   Quantidade de dados de entrada
        n_hidden:   Quantidade de neurônios nas camadas ocultas
        n_layers:   Quantidade de camadas
        activation: Função de ativação

    Retorna: (theta, x_ddot) — ângulo e aceleração fisicamente consistentes
    """
    def __init__(self, n_inputs, n_hidden, n_layers, activation):
        super().__init__()

        # Rede de política — aprende θ
        layers = []
        layers.append(nn.Linear(n_inputs, n_hidden))
        layers.append(activation())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activation())

        layers.append(nn.Linear(n_hidden, 1))
        self.policy_net = nn.Sequential(*layers)

        # Constantes físicas como buffers — não são parâmetros treináveis
        self.register_buffer('alpha', torch.tensor(ALPHA, dtype=torch.float32))
        self.register_buffer('g',     torch.tensor(G,     dtype=torch.float32))

    def forward(self, X):
        """
        Entrada: [x, ẋ, x_ref, e_int]
        Saída:   theta (ângulo comandado em graus)

        A aceleração ẍ = (5/9)·g·θ é satisfeita por construção
        no simulador diferenciável.
        """
        theta = self.policy_net(X)
        return theta

    def physics(self, theta):
        """
        Calcula ẍ a partir de θ pela equação física — hard constraint.
        Nunca pode ser violada independente dos pesos da rede.
        """
        return self.alpha * self.g * theta