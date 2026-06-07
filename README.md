# Ball and Beam — Controle por Rede Neural Embarcada

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-ESP32--S3-00599C?logo=cplusplus&logoColor=white)
![PlatformIO](https://img.shields.io/badge/PlatformIO-Firmware-orange?logo=platformio&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Search-6C63FF?logo=optuna&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Implementação de um controlador neural para o sistema **Ball and Beam**, treinado por **simulação** e embarcado em um **ESP32-S3**.

A rede aprende uma política de controle π(x, ẋ, x_ref, e_int) → θ inteiramente *offline*, sem dados experimentais rotulados. A física do sistema não é uma penalização externa — ela é o próprio simulador que evolui os estados durante o treinamento.

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
├── docs/                         # Arquivos do relatório do projeto e vídeos da validação online
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
| Estrutura           | Impressão 3D — PLA|
| Bola                | Tênis de mesa     |

O esquema elétrico está contido no relatório em \docs
---

## Instalação

**Pré-requisitos:** Python 3.10+, CUDA opcional (treino roda em CPU também)

```bash
git clone https://github.com/EdelioGabriel/BALL_AND_BEAM.git
cd BALL_AND_BEAM/pinn
pip install -r requirements
```

Para o firmware, indica-se que instale a extensão [PlatformIO](https://platformio.org/) no VSCode, e configure um ambiente de projeto com suporte ao ESP32-S3.

---

## Fluxo completo

### 1. Coleta de dados (opcional)

A coleta é opcional — o treino não depende de dados reais. Use para monitorar os sinais de controle em tempo real e salvar os dados para análise posterior.

```bash
python scripts/generate_data.py                          # Linux/Mac (auto-detecta porta)
```

Comandos disponíveis via terminal durante a coleta:

| Comando | Ação                      |
|---------|---------------------------|
| `A`     | Ativa o controlador       |
| `D`     | Desativa o controlador    |
| `R`     | Reset do sistema          |
| `P`     | Alterna entre LikePINN e controlador clássico |
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

Vale ressaltar que a otimização foi feita utilizando GPU (RTX4090), devido ao tempo de execução. Caso deseje realizar seu próprio estudo, esteja ciente que pode demorar muitos minutos.

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
O código ainda não copia a conversão automaticamente, por isso deve ser manual.

Compile e faça upload com PlatformIO.

---

Os demais detalhes sobre a implementação da rede neural estão descritos no relatório na pasta \docs.
---

## Firmware e inferência embarcada

A inferência no ESP32-S3 é implementada manualmente — sem TFLite ou qualquer biblioteca externa (futura implementação). Os pesos são carregados de `likepinn_weights.h` como arrays C e a forward pass é uma sequência de multiplicações matriciais com SiLU aplicada elemento a elemento.

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
- Lin, J., Zhu, L., Chen, W.-M., Wang, W.-C., Han, S. (2024). *Tiny Machine Learning: Progress and Futures*. [arXiv:2403.19076](https://arxiv.org/abs/2403.19076)
- Da Silva Neto, E. (2021). *TinyML: Machine learning para microcontroladores*. Embarcados. [https://embarcados.com.br/tinyml-machine-learning-para-microcontroladores/](https://embarcados.com.br/tinyml-machine-learning-para-microcontroladores/)
- Ogata, K. (2010). *Engenharia de Controle Moderno* (5ª ed.). Pearson.

## Agradecimentos

Agradeço ao meu irmão, Wallace Magalhães, exímio Engenheiro Eletricista que me guiou por essa jornada em sistema de controle, além de me disponibilizar o código para o controle clássico, sem o qual certamente enfrentaria muitos percalços para implementar.
