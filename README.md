# Ball and Beam — Controle por Rede Neural Embarcada

Implementação de um controlador baseado em rede neural (*PolicyNet*) para o sistema *Ball and Beam*, treinado por **simulação diferenciável** e embarcado em um microcontrolador **ESP32-S3** sem bibliotecas de inferência externas.

---

## Visão geral

O controlador aprende a política π(x, ẋ, x_ref, e_int) → θ inteiramente *offline*, sem dados experimentais rotulados. A física do sistema é incorporada no próprio simulador que gera as trajetórias de treinamento — não como resíduo penalizado, mas como dinâmica que propaga os estados.

```
Equação do movimento (linearizada):
    ẍ = (5/9) · g · θ
```

Os pesos treinados são exportados como um header C (`likepinn_weights.h`) e a inferência é implementada manualmente no firmware — multiplicações matriciais em ponto flutuante, sem TFLite.

---

## Estrutura do projeto

```
.
├── scripts/
│   ├── skeleton.py            # Arquiteturas: PINNIdentifier e PolicyNet
│   ├── train.py               # Simulador diferenciável + loop de treino
│   ├── optimization.py        # Busca de hiperparâmetros com Optuna
│   ├── final_train.py         # Treino final com melhores hiperparâmetros
│   ├── export_model.py        # Exporta pesos para header C + validação
│   └── generate_data.py       # Coleta serial do ESP32 com plot em tempo real
├── export/
│   └── likepinn_weights.h     # Pesos exportados (gerado automaticamente)
├── data/                      # CSVs coletados do hardware
└── firmware/
    └── main.cpp               # Firmware ESP32-S3
```

---

## Fluxo completo

### 1. Coleta de dados (opcional)

Conecte o ESP32-S3 e rode com qualquer controlador ativo (clássico ou manual):

```bash
python scripts/generate_data.py
python scripts/generate_data.py COM3 115200          # Windows
python scripts/generate_data.py /dev/ttyUSB0 115200  # Linux/Mac
```

Comandos disponíveis via terminal durante a coleta:

| Comando | Ação |
|---|---|
| `A` | Ativa o controlador |
| `D` | Desativa o controlador |
| `R` | Reset do sistema |
| `C` | Liga/desliga gravação CSV |
| `15.0` | Define novo setpoint (cm) |
| `q` | Encerra o script |

### 2. Otimização de hiperparâmetros

```bash
python scripts/optimization.py
```

Os resultados são salvos em `scripts/optuna/policynet_study.db` e podem ser consultados com o dashboard do Optuna:

```bash
optuna-dashboard sqlite:///scripts/optuna/policynet_study.db
```

### 3. Treino final

```bash
python scripts/final_train.py
```

Salva o modelo em `scripts/results/likepinn_best.pth` e gera gráficos de validação.

### 4. Exportação para C

```bash
python scripts/export_model.py
```

Gera `export/likepinn_weights.h` e imprime a validação numérica:

```
Validação manual vs PyTorch:
  PyTorch: 12.345678
  Manual:  12.345679
  Erro:    1.23e-07
  ✓ Equivalência confirmada
```

### 5. Firmware

Copie o header gerado para o firmware e compile:

```bash
cp export/likepinn_weights.h firmware/include/
```

---

## Arquitetura da rede

| Parâmetro | Valor |
|---|---|
| Entradas | 4 — `[x, ẋ, x_ref, e_int]` |
| Camadas ocultas | 3 × 256 neurônios |
| Ativação | SiLU |
| Saída | 1 — θ (graus), saturado em ±60° |
| Parâmetros totais | ~198 mil |

### Hiperparâmetros (resultado do Optuna)

| Hiperparâmetro | Valor |
|---|---|
| *batch size* | 316 |
| *learning rate* | 10⁻³ |
| w_state | 5.11 |
| w_effort | 0.35 |
| Ativação | SiLU |

---

## Função de perda

$$\mathcal{L} = w_{\text{state}} \cdot \mathcal{L}_{\text{state}} + w_{\text{effort}} \cdot \mathcal{L}_{\text{effort}}$$

Onde:
- **L_state** — MSE entre a trajetória simulada e o setpoint ao longo de todos os passos
- **L_effort** — MSE dos ângulos comandados (penaliza ações excessivas)

---

## Hardware

| Componente | Modelo |
|---|---|
| Microcontrolador | ESP32-S3 |
| Sensor de distância | VL53L0X (laser) |
| Atuador | Servomotor MG996R |
| Estrutura | Impressão 3D — PLA (GrabCAD) |
| Bola | Tênis de mesa |

---

## Comandos do firmware

| Comando (serial) | Ação |
|---|---|
| `A` | Ativa o controlador |
| `D` | Desativa o controlador |
| `R` | Reset (zera integrador e setpoint) |
| `P` | Alterna entre LikePINN e clássico |
| `C` | Liga/desliga modo coleta CSV |
| `15.0` | Define novo setpoint em cm |

---

## Dependências Python

```bash
pip install torch numpy pandas matplotlib optuna pyserial
```

---

## Referências

- Raissi, M. et al. (2017). *Physics Informed Deep Learning*. [arXiv:1711.10561](https://arxiv.org/abs/1711.10561)
- Baty, H. (2024). *A Practical Introduction to Physics-Informed Neural Networks*. [arXiv:2403.00599](https://arxiv.org/abs/2403.00599)
