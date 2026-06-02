"""
Exporta os pesos da LikePINN diretamente como array C para o ESP32-S3.
Implementação manual da inferência — sem TFLite.
"""

import sys
import re
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.skeleton import LikePINN

DEVICE      = torch.device('cpu')
RESULTS_DIR = Path(__file__).parent / 'results'
EXPORT_DIR  = Path(__file__).parent.parent / 'export'
EXPORT_DIR.mkdir(exist_ok=True)

def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    config     = checkpoint['config']

    activation_map = {'Tanh': nn.Tanh, 'SiLU': nn.SiLU}

    model = LikePINN(
        n_inputs   = config['n_inputs'],
        n_outputs  = config['n_outputs'],
        n_hidden   = config['n_hidden'],
        n_layers   = config['n_layers'],
        activation = activation_map[config['activation']]
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config

def export_c_header(model, config, path):
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.data.numpy()

    lines = []
    lines.append("/**")
    lines.append(" * Pesos da LikePINN Ball and Beam — gerado automaticamente")
    lines.append(f" * Arquitetura: {config['n_inputs']} → {config['n_hidden']} × {config['n_layers']} → {config['n_outputs']}")
    lines.append(f" * Ativação: {config['activation']} | NÃO EDITAR MANUALMENTE")
    lines.append(" */")
    lines.append("")
    lines.append("#ifndef LikePINN_WEIGHTS_H")
    lines.append("#define LikePINN_WEIGHTS_H")
    lines.append("")
    lines.append(f"#define LikePINN_N_INPUTS  {config['n_inputs']}")
    lines.append(f"#define LikePINN_N_HIDDEN  {config['n_hidden']}")
    lines.append(f"#define LikePINN_N_LAYERS  {config['n_layers']}")
    lines.append(f"#define LikePINN_N_OUTPUTS {config['n_outputs']}")
    lines.append("")

    for name, w in weights.items():
        var_name = name.replace('.', '_')
        flat     = w.flatten()
        shape    = w.shape
        vals     = ', '.join(f'{v:.8f}f' for v in flat)
        lines.append(f"// {name} — shape {shape}")
        lines.append(f"const float {var_name}[] = {{{vals}}};")
        lines.append("")

    lines.append("#endif // LikePINN_WEIGHTS_H")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Header exportado: {path}")
    print(f"Tamanho: {path.stat().st_size} bytes")

def validate(model, config):
    """Testa inferência manual contra PyTorch para garantir equivalência."""

    header_path = EXPORT_DIR / 'likepinn_weights.h'
    with open(header_path, encoding='utf-8') as f:
        content = f.read()

    # Extrai arrays do header
    arrays = {}
    for match in re.finditer(r'const float (\w+)\[\] = \{([^}]+)\}', content):
        name = match.group(1)
        vals = np.array([float(v.strip().rstrip('f')) for v in match.group(2).split(',')])
        arrays[name] = vals

    # Função de ativação correta
    act_fn = np.tanh if config['activation'] == 'Tanh' \
             else lambda x: x / (1 + np.exp(-x))

    # Inferência manual
    x = np.random.randn(config['n_inputs']).astype(np.float32)
    h = x
    for i in range(config['n_layers']):
        idx  = f'net_{i*2}_weight'
        bias = f'net_{i*2}_bias'
        W    = arrays[idx].reshape(config['n_hidden'], -1 if i == 0 else config['n_hidden'])
        b    = arrays[bias]
        h    = act_fn(W @ h + b)

    # Camada de saída
    W_out    = arrays[f'net_{config["n_layers"]*2}_weight'].reshape(config['n_outputs'], config['n_hidden'])
    b_out    = arrays[f'net_{config["n_layers"]*2}_bias']
    y_manual = (W_out @ h + b_out)[0]

    # PyTorch
    with torch.no_grad():
        y_torch = model(torch.tensor(x).unsqueeze(0)).item()

    erro = abs(y_manual - y_torch)
    print(f"\nValidação manual vs PyTorch:")
    print(f"  PyTorch: {y_torch:.6f}")
    print(f"  Manual:  {y_manual:.6f}")
    print(f"  Erro:    {erro:.2e}")
    print(f"  {'✓ Equivalência confirmada' if erro < 1e-5 else '✗ Erro alto'}")

if __name__ == '__main__':
    model_path  = RESULTS_DIR / 'likepinn.pth'
    header_path = EXPORT_DIR  / 'likepinn_weights.h'

    print("=" * 50)
    print("Exportando pesos da LikePINN para C")
    print("=" * 50)

    model, config = load_model(model_path)
    print(f"Modelo: {config}")

    export_c_header(model, config, header_path)
    validate(model, config)

    print("\n" + "=" * 50)
    print(f"Copie {header_path} para firmware/include/")
    print("=" * 50)