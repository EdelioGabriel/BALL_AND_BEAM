"""
Captura serial interativa com plot em tempo real — Ball and Beam
----------------------------------------------------------------
- Thread de leitura: grava CSV e exibe debug no terminal
- Thread de comando: você digita A / D / R / C / 15.0 etc. direto aqui
- Plot em tempo real: posição, setpoint e sinal de controle

Uso:
    python generate_data.py
    python generate_data.py COM3 115200          # Windows
    python generate_data.py /dev/ttyUSB0 115200  # Linux/Mac

Comandos disponíveis (mesmos do firmware):
    A       — ativar controlador
    D       — desativar controlador
    R       — reset (também reseta e_int aqui no script)
    C       — toggle coleta CSV
    15.0    — novo setpoint (qualquer número positivo)
    q       — encerrar o script
"""

import serial
import sys
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Estilo científico ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    13,
    'legend.fontsize':   10,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.grid':         True,
    'grid.linestyle':    '--',
    'grid.alpha':        0.4,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        120,
})

# ── Configuração padrão ───────────────────────────────────────────────
DEFAULT_PORT     = "COM19"
DEFAULT_BAUDRATE = 115200
OUTPUT_DIR       = Path(__file__).parent.parent / "data"
DT               = 0.05
CSV_HEADER       = "tempo_s,posicao_cm,velocidade_cms,x_ref_cm,e_int,u_K\n"
JANELA_PLOT      = 200   # número de amostras visíveis no plot (~10s)
# ─────────────────────────────────────────────────────────────────────

# Estado compartilhado entre threads
amostras   = 0
coletando  = False
stop_event = threading.Event()
file_lock  = threading.Lock()

state_lock = threading.Lock()
x_ref      = 0.0
e_int      = 0.0

# Buffers para o plot em tempo real
plot_lock  = threading.Lock()
buf_tempo  = deque(maxlen=JANELA_PLOT)
buf_pos    = deque(maxlen=JANELA_PLOT)
buf_ref    = deque(maxlen=JANELA_PLOT)
buf_u      = deque(maxlen=JANELA_PLOT)


def is_csv_line(line: str) -> bool:
    parts = line.strip().split(",")
    if len(parts) != 4:
        return False
    try:
        [float(p) for p in parts]
        return True
    except ValueError:
        return False


def thread_leitura(ser, filepath):
    global amostras, coletando, e_int

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(CSV_HEADER)

        while not stop_event.is_set():
            try:
                raw = ser.readline()
            except Exception:
                break

            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            if not line:
                continue

            if is_csv_line(line):
                parts = line.strip().split(",")
                try:
                    tempo   = float(parts[0])
                    posicao = float(parts[1])
                    u_k     = float(parts[3])
                except ValueError:
                    continue

                with state_lock:
                    current_x_ref = x_ref
                    erro = current_x_ref - posicao
                    e_int += erro * DT
                    current_e_int = e_int

                enriched = f"{parts[0]},{parts[1]},{parts[2]},{current_x_ref:.4f},{current_e_int:.4f},{parts[3]}"

                with file_lock:
                    f.write(enriched + "\n")
                    f.flush()

                with plot_lock:
                    buf_tempo.append(tempo)
                    buf_pos.append(posicao)
                    buf_ref.append(current_x_ref)
                    buf_u.append(u_k)

                amostras += 1
                coletando = True

                if amostras % 50 == 0:
                    print(f"\n  [{amostras} amostras] pos={posicao:.2f}cm ref={current_x_ref:.1f}cm u={u_k:.2f}°")
                    print("  cmd> ", end="", flush=True)
            else:
                if "COLETA" in line.upper():
                    coletando = "ON" in line.upper()
                print(f"\n  >> {line}")
                print("  cmd> ", end="", flush=True)


def thread_comandos(ser):
    global x_ref, e_int

    print("\n  Comandos: A D R C  |  número = novo setpoint  |  q = sair")
    print("  " + "-" * 50)
    print(f"  Setpoint inicial: {x_ref} cm")

    while not stop_event.is_set():
        try:
            print("  cmd> ", end="", flush=True)
            cmd = input().strip()
        except EOFError:
            break

        if not cmd:
            continue

        if cmd.lower() == "q":
            stop_event.set()
            break

        try:
            novo_ref = float(cmd)
            if 0.0 <= novo_ref <= 20.0:
                with state_lock:
                    x_ref = novo_ref
                    e_int = 0.0
                print(f"  Setpoint atualizado: {novo_ref} cm | e_int resetado")
            else:
                print("  Setpoint fora da faixa (0 a 20 cm) — ignorado")
        except ValueError:
            if cmd.upper() == "R":
                with state_lock:
                    e_int = 0.0
                print("  e_int resetado localmente")

        try:
            ser.write((cmd + "\n").encode("utf-8"))
        except Exception as e:
            print(f"  Erro ao enviar: {e}")


def plot_tempo_real(filepath):
    """Plot em tempo real com estilo científico."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08}
    )
    fig.suptitle("Ball and Beam — Monitoramento em Tempo Real", fontweight='bold', y=0.98)

    # Painel superior — posição
    line_pos, = ax1.plot([], [], color='#2166ac', lw=1.5, label=r'$x(t)$ — posição')
    line_ref, = ax1.plot([], [], color='#d73027', lw=1.2, linestyle='--', label=r'$x_{ref}$ — setpoint')
    ax1.set_ylabel('Posição (cm)')
    ax1.set_ylim(-1, 32)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_title('')

    # Painel inferior — sinal de controle
    line_u, = ax2.plot([], [], color='#1a9641', lw=1.5, label=r'$\theta(t)$ — ângulo comandado')
    ax2.axhline(0, color='k', lw=0.8, linestyle=':')
    ax2.set_ylabel('Ângulo (graus)')
    ax2.set_xlabel('Tempo (s)')
    ax2.legend(loc='upper right', framealpha=0.9)

    def update(frame):
        with plot_lock:
            if len(buf_tempo) < 2:
                return line_pos, line_ref, line_u

            t   = list(buf_tempo)
            pos = list(buf_pos)
            ref = list(buf_ref)
            u   = list(buf_u)

        line_pos.set_data(t, pos)
        line_ref.set_data(t, ref)
        line_u.set_data(t, u)

        ax1.set_xlim(t[0], t[-1] + 0.5)
        ax2.set_xlim(t[0], t[-1] + 0.5)

        u_max = max(abs(min(u)), abs(max(u))) + 5
        ax2.set_ylim(-u_max, u_max)

        return line_pos, line_ref, line_u

    ani = animation.FuncAnimation(
        fig, update,
        interval         = 200,
        blit             = True,
        cache_frame_data = False
    )

    plt.tight_layout()

    def on_close(event):
        save_path = filepath.replace('.csv', '.png')
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n  Plot salvo: {save_path}")

    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

    stop_event.set()


def main():
    port     = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT
    baudrate = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BAUDRATE

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = str(OUTPUT_DIR / f"coleta_{timestamp}.csv")

    print(f"\n  Porta:   {port} @ {baudrate} baud")
    print(f"  Arquivo: {filepath}")
    print(f"  DT:      {DT}s | Janela: {JANELA_PLOT * DT:.0f}s")
    print("  Conectando...")

    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except serial.SerialException as e:
        print(f"\n  Erro: {e}")
        print("  Verifique a porta e se o ESP32 está conectado.")
        sys.exit(1)

    time.sleep(2)

    t_leitura  = threading.Thread(target=thread_leitura,  args=(ser, filepath), daemon=True)
    t_comandos = threading.Thread(target=thread_comandos, args=(ser,),          daemon=True)

    t_leitura.start()
    t_comandos.start()

    plot_tempo_real(filepath)

    t_leitura.join(timeout=2)
    ser.close()

    print(f"\n  Encerrado. Total de amostras: {amostras}")
    print(f"  Arquivo salvo: {filepath}\n")


if __name__ == "__main__":
    main()