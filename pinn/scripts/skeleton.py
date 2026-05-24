"""
Este script contém todas as funções e estruturas gerais para a implementação dos exemplos, evitando repetição desnecessária

- Nota do autor
"""

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

# Classe de construção da arquitetura da rede - tipo MLP (Multilayer Perceptron)
class PINN(nn.Module):
    """
    Classe para definir a arquitetura MLP
    Recebe como parâmetros de arquitetura:

        n_inputs: Quantidade de dados de entrada
        n_outputs: Quantidade de dados de saída
        n_hidden: Quantidade de neurônios nas camadas ocultas
        n_layers: Quantidade de camadas
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
        X = self.net(X)
        return X
  