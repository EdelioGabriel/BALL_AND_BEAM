# Ball and Beam — Controle por Rede Neural Embarcada

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-ESP32--S3-00599C?logo=cplusplus&logoColor=white)
![PlatformIO](https://img.shields.io/badge/PlatformIO-Firmware-orange?logo=platformio&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Search-6C63FF?logo=optuna&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Implementação de um controlador neural para o sistema **Ball and Beam**, treinado por **simulação diferenciável** e embarcado em um **ESP32-S3** sem bibliotecas de inferência externas.

A rede aprende uma política de controle π(x, ẋ, x_ref, e_int) → θ inteiramente *offline*, sem dados experimentais rotulados. A física do sistema não é uma penalização externa — ela é o próprio simulador que propaga os estados durante o treinamento.

---

## Sumário

- [Motivação](#motivação)
- [Como funciona](#como-funciona)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Hardware](#hardware)
- [Instalação](#instalação)
- [Fluxo completo](#fluxo-completo)
- [Arquitetura da rede](#arquitetura-da-rede)
- [Função de perda](#função-de-perda)
- [Simulador diferenciável](#simulador-diferenciável)
- [Hiperparâmetros](#hiperparâmetros)
- [Firmware e inferência embarcada](#firmware-e-inferência-embarcada)
- [Referências](#referências)

---

## Motivação

Controladores clássicos (PID, LQR) dependem de um modelo linear preciso e são sensíveis a não-linearidades e perturbações. Este projeto explora uma alternativa: treinar uma rede neural diretamente sobre a dinâmica do sistema, usando o simulador físico como parte do grafo de computação. O resultado é uma política que generaliza para diferentes condições iniciais e setpoints sem nunca ter visto dados reais de hardware durante o treino.

---

## Como funciona

O treinamento ocorre em três etapas por época:

1. **Amostragem** — condições iniciais (x₀, ẋ₀, x_ref) são sorteadas aleatoriamente no espaço de operação físico
2. **Simulação diferenciável** — a rede produz θ(t) a cada passo; a EDO do sistema propaga os estados; gradientes fluem de volta pela trajetória inteira
3. **Otimização** — a loss penaliza desvio do setpoint, esforço de controle e resíduo físico

Após o treino, os pesos são exportados como um header C e a inferência roda no microcontrolador com multiplicações matriciais simples em ponto flutuante.

```
Equação de movimento (linearizada):
    ẍ = (5/9) · g · θ
```

---

## Estrutura do projeto

```
BALL_AND_BEAM/
├── firmware/
│   └── src/
│       └── main.cpp              # Firmware ESP32-S3 (inferência + controle)
│   └── include/
│       └── likepinn_weights.h    # Pesos exportados (copiado após export)
├── pinn/
│   └── scripts/
│       ├── skeleton.py           # Arquitetura LikePINN (MLP)
│       ├── train.py              # Simulador diferenciável + loop de treino
│       ├── optimization.py       # Busca de hiperparâmetros com Optuna
│       ├── final_train.py        # Treino final com melhores hiperparâmetros
│       ├── export_model.py       # Exporta pesos para header C + validação
│       └── generate_data.py      # Coleta serial do ESP32 com plot em tempo real
│   └── export/
│       └── likepinn_weights.h    # Gerado automaticamente pelo export_model.py
│   └── data/                     # CSVs coletados do hardware
│   └── optuna/
│       └── policynet_study.db    # Banco de dados do Optuna
├── docs/
├── BALL_AND_BEAM.code-workspace
└── README.md
```

---

## Hardware

| Componente          | Modelo            |
|---------------------|-------------------|
| Microcontrolador    | ESP32-S3          |
| Sensor de distância | VL53L0X (laser)   |
| Atuador             | Servomotor MG996R |
| Estrutura           | Impressão 3D — PLA (GrabCAD) |
| Bola                | Tênis de mesa     |

---

## Instalação

**Pré-requisitos:** Python 3.10+, CUDA opcional (treino roda em CPU também)

```bash
git clone https://github.com/EdelioGabriel/BALL_AND_BEAM.git
cd BALL_AND_BEAM/pinn
pip install torch numpy pandas matplotlib optuna pyserial
```

Para o firmware: [PlatformIO](https://platformio.org/) com suporte ao ESP32-S3.

---

## Fluxo completo

### 1. Coleta de dados (opcional)

A coleta é opcional — o treino não depende de dados reais. Use para validação ou behavior cloning futuro.

```bash
python scripts/generate_data.py                          # Linux/Mac (auto-detecta porta)
python scripts/generate_data.py COM3 115200              # Windows
python scripts/generate_data.py /dev/ttyUSB0 115200      # Linux/Mac (explícito)
```

Comandos disponíveis via terminal durante a coleta:

| Comando | Ação                      |
|---------|---------------------------|
| `A`     | Ativa o controlador       |
| `D`     | Desativa o controlador    |
| `R`     | Reset do sistema          |
| `C`     | Liga/desliga gravação CSV |
| `15.0`  | Define novo setpoint (cm) |
| `q`     | Encerra o script          |

### 2. Busca de hiperparâmetros

```bash
python scripts/optimization.py
```

Resultados salvos em `optuna/policynet_study.db`. Para visualizar o dashboard:

```bash
optuna-dashboard sqlite:///optuna/policynet_study.db
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

Gera `export/likepinn_weights.h` e valida a equivalência numérica entre PyTorch e a implementação manual:

```
Validação manual vs PyTorch:
  PyTorch: 12.345678
  Manual:  12.345679
  Erro:    1.23e-07
  ✓ Equivalência confirmada
```

### 5. Deploy no firmware

```bash
cp export/likepinn_weights.h firmware/include/
```

Compile e faça upload com PlatformIO.

---

## Arquitetura da rede

A `LikePINN` é uma MLP com estrutura configurável, instanciada para o problema de controle como:

| Parâmetro       | Valor                              |
|-----------------|------------------------------------|
| Entradas        | 4 — `[x, ẋ, x_ref, e_int]`        |
| Camadas ocultas | 3 × 256 neurônios                  |
| Ativação        | SiLU                               |
| Saída           | 1 — θ (graus), saturado em ±60°   |
| Parâmetros      | ~198 mil                           |

O vetor de entrada captura o estado completo necessário para uma ação de controle: posição atual, velocidade, referência desejada e erro acumulado (termo integral).

---

## Função de perda

$$\mathcal{L} = w_{\text{state}} \cdot \mathcal{L}_{\text{state}} + w_{\text{effort}} \cdot \mathcal{L}_{\text{effort}} + w_{\text{edo}} \cdot \mathcal{L}_{\text{edo}}$$

| Termo | Descrição |
|-------|-----------|
| **L_state** | MSE entre trajetória simulada e setpoint em todos os passos |
| **L_effort** | MSE dos ângulos comandados — penaliza ações excessivas |
| **L_edo** | Resíduo da EDO — penaliza trajetórias fisicamente inconsistentes |

O resíduo físico é calculado estimando ẍ por diferenças finitas na trajetória simulada e comparando com `(5/9)·g·θ`:

```python
x_ddot_sim = torch.diff(torch.diff(xs, dim=1), dim=1) / (DT ** 2)
residual   = x_ddot_sim - ALPHA * G * theta_mid
loss_edo   = torch.mean(residual ** 2)
```

---

## Simulador diferenciável

A função `simulate()` rola a política da rede por `n_steps` passos usando integração de Euler explícita:

```
e_int ← e_int + (x_ref − x) · Δt
θ     ← clamp(π(x, ẋ, x_ref, e_int), −60°, 60°)
ẍ     ← (5/9) · g · θ
ẋ     ← ẋ + ẍ · Δt
x     ← clamp(x + ẋ · Δt, 0, 30 cm)
```

Como todas as operações são diferenciáveis (PyTorch), o gradiente da loss flui de volta pela trajetória inteira até os pesos da rede.

**Parâmetros físicos do simulador:**

| Parâmetro    | Valor   |
|--------------|---------|
| Δt           | 0.05 s  |
| x ∈           | [0, 30] cm |
| ẋ_max        | 10 cm/s |
| θ_max        | 60°     |

---

## Hiperparâmetros

Resultado da busca com Optuna:

| Hiperparâmetro | Valor |
|----------------|-------|
| Batch size     | 316   |
| Learning rate  | 10⁻³  |
| w_state        | 5.11  |
| w_effort       | 0.35  |
| Ativação       | SiLU  |

---

## Firmware e inferência embarcada

A inferência no ESP32-S3 é implementada manualmente — sem TFLite ou qualquer biblioteca externa. Os pesos são carregados de `likepinn_weights.h` como arrays C e a forward pass é uma sequência de multiplicações matriciais com SiLU aplicada elemento a elemento.

Comandos disponíveis via serial durante operação:

| Comando | Ação                                  |
|---------|---------------------------------------|
| `A`     | Ativa o controlador                   |
| `D`     | Desativa o controlador                |
| `R`     | Reset (zera integrador e setpoint)    |
| `P`     | Alterna entre LikePINN e controlador clássico |
| `C`     | Liga/desliga modo coleta CSV          |
| `15.0`  | Define novo setpoint (cm)             |

---

## Referências

- Raissi, M., Perdikaris, P., Karniadakis, G. E. (2017). *Physics Informed Deep Learning*. [arXiv:1711.10561](https://arxiv.org/abs/1711.10561)
- Baty, H. (2024). *A Practical Introduction to Physics-Informed Neural Networks*. [arXiv:2403.00599](https://arxiv.org/abs/2403.00599)
