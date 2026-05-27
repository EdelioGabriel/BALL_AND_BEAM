#!/bin/bash
#SBATCH --job-name=ball_and_beam
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=work6
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edelio25024@ilum.cnpem.br
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

# Diretório base do projeto
SCRIPTS_DIR=/home/edelio25024/work/PINN_BALL_AND_BEAM/BALL_AND_BEAM/pinn/scripts

cd "$SCRIPTS_DIR"

# Cria pastas de output se não existirem
mkdir -p logs
mkdir -p optuna
mkdir -p ../tests/results

echo "======================================"
echo "Job iniciado em: $(date)"
echo "Diretório: $(pwd)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "======================================"

# Ativa o venv
source /home/edelio25024/work/PINN_BALL_AND_BEAM/BALL_AND_BEAM/pinn/.venv/bin/activate

echo "Python: $(python -c 'import sys; print(sys.executable)')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA disponível: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "======================================"

echo "Iniciando otimização de hiperparâmetros..."

python export_model.py

echo "======================================"
echo "Finalizado em: $(date)"
echo "======================================"