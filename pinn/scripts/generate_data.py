"""
Captura serial interativa — Ball and Beam
-----------------------------------------
- Thread de leitura: grava CSV e exibe debug no terminal
- Thread de comando: você digita A / D / R / C / 15.0 etc. direto aqui
- Não precisa do monitor serial do VS Code

Uso:
    pip install pyserial
    python captura_serial.py                      # porta padrão abaixo
    python captura_serial.py COM3 115200          # Windows
    python captura_serial.py /dev/ttyUSB0 115200  # Linux/Mac

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

# ── Configuração padrão ───────────────────────────────────────────────
DEFAULT_PORT     = "COM19"
DEFAULT_BAUDRATE = 115200
OUTPUT_DIR       = "data"
DT               = 0.05   # período de amostragem em segundos
CSV_HEADER       = "tempo_s,posicao_cm,velocidade_cms,x_ref_cm,e_int,u_K\n"
# ─────────────────────────────────────────────────────────────────────

# Estado compartilhado entre threads
amostras   = 0
coletando  = False
stop_event = threading.Event()
csv_file   = None
file_lock  = threading.Lock()

# Estado do controlador (compartilhado entre threads)
state_lock = threading.Lock()
x_ref      = 0.0   # setpoint atual em cm
e_int      = 0.0   # erro integrado acumulado


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
    """Lê a serial continuamente, separa CSV de debug, salva e exibe."""
    global amostras, coletando, csv_file, e_int

    with open(filepath, "w") as f:
        f.write(CSV_HEADER)
        csv_file = f

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
                    posicao = float(parts[1])
                except ValueError:
                    continue

                # Atualiza e_int com o erro atual
                with state_lock:
                    current_x_ref = x_ref
                    erro = current_x_ref - posicao
                    e_int += erro * DT
                    current_e_int = e_int

                # Monta linha enriquecida
                enriched = f"{parts[0]},{parts[1]},{parts[2]},{current_x_ref:.4f},{current_e_int:.4f},{parts[3]}"

                with file_lock:
                    f.write(enriched + "\n")
                    f.flush()

                amostras += 1
                coletando = True

                if amostras % 50 == 0:
                    print(f"\n  [{amostras} amostras] {enriched}")
                    print("  cmd> ", end="", flush=True)
            else:
                # detecta se o firmware confirmou toggle do modo coleta
                if "COLETA" in line.upper():
                    coletando = "ON" in line.upper()
                print(f"\n  >> {line}")
                print("  cmd> ", end="", flush=True)


def thread_comandos(ser):
    """Lê comandos do teclado e envia para o ESP32."""
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

        # Tenta interpretar como novo setpoint
        try:
            novo_ref = float(cmd)
            if 0.0 <= novo_ref <= 20.0:
                with state_lock:
                    x_ref = novo_ref
                    e_int = 0.0   # reseta integrador ao mudar setpoint
                print(f"  Setpoint atualizado: {novo_ref} cm | e_int resetado")
            else:
                print("  Setpoint fora da faixa (0 a 20 cm) — ignorado")
        except ValueError:
            # Não é número — trata como comando de firmware
            if cmd.upper() == "R":
                with state_lock:
                    e_int = 0.0
                print("  e_int resetado localmente")

        # Envia para o ESP32 com newline
        try:
            ser.write((cmd + "\n").encode("utf-8"))
        except Exception as e:
            print(f"  Erro ao enviar: {e}")


def main():
    port     = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT
    baudrate = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BAUDRATE

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = os.path.join(OUTPUT_DIR, f"coleta_{timestamp}.csv")

    print(f"\n  Porta:   {port} @ {baudrate} baud")
    print(f"  Arquivo: {filepath}")
    print(f"  DT:      {DT}s | Colunas: {CSV_HEADER.strip()}")
    print("  Conectando...")

    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except serial.SerialException as e:
        print(f"\n  Erro: {e}")
        print("  Verifique a porta e se o ESP32 está conectado.")
        sys.exit(1)

    time.sleep(2)  # aguarda ESP32 reiniciar

    t_leitura  = threading.Thread(target=thread_leitura,  args=(ser, filepath), daemon=True)
    t_comandos = threading.Thread(target=thread_comandos, args=(ser,),          daemon=True)

    t_leitura.start()
    t_comandos.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()

    t_leitura.join(timeout=2)

    ser.close()
    print(f"\n  Encerrado. Total de amostras: {amostras}")
    print(f"  Arquivo salvo: {filepath}\n")


if __name__ == "__main__":
    main()