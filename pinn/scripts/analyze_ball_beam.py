"""
Ball and Beam — Análise de Métricas de Controle
================================================
Detecta automaticamente as transições de setpoint e calcula,
para cada degrau:
  • Tempo de subida (10%→90%)
  • Tempo de acomodação (critério ±2% e ±5%)
  • Overshoot (%)
  • Undershoot (%)
  • Erro em regime permanente
  • Valor de pico

Uso:
    python analyze_ball_beam.py                          # busca CSV mais recente em ../data/
    python analyze_ball_beam.py caminho/para/arquivo.csv
"""

import sys
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker

# ── Estilo científico ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'legend.fontsize':    9,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.grid':         True,
    'grid.linestyle':    '--',
    'grid.alpha':        0.35,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        130,
})

CORES = ['#2166ac', '#d73027', '#1a9641', '#8856a7', '#e08214']

# ── Tolerâncias para tempo de acomodação ─────────────────────────────
TOL_2  = 0.02   # ±2 %
TOL_5  = 0.05   # ±5 %

# ── Janela mínima de amostras após degrau ─────────────────────────────
MIN_STEP_SAMPLES = 30   # ignora transições muito curtas


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def find_csv() -> Path:
    """Encontra o CSV mais recente na pasta data/ padrão do projeto."""
    # Caminho padrão conforme generate_data.py: pasta pai do script / data /
    script_dir = Path(__file__).resolve().parent
    data_dir   = script_dir.parent / "data"

    # Tenta também na mesma pasta do script
    candidates = sorted(
        list(data_dir.glob("coleta_*.csv")) +
        list(script_dir.glob("coleta_*.csv")) +
        list(script_dir.glob("*.csv")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {data_dir} nem em {script_dir}.\n"
            "Passe o caminho explícito: python analyze_ball_beam.py seu_arquivo.csv"
        )
    return candidates[0]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"tempo_s", "posicao_cm", "x_ref_cm", "u_K"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
    df = df.sort_values("tempo_s").reset_index(drop=True)
    return df


def detect_steps(df: pd.DataFrame) -> list[dict]:
    """
    Detecta onde x_ref muda de valor e retorna lista de segmentos.
    Cada segmento: {'ref': float, 'i_start': int, 'i_end': int,
                    'ref_prev': float, 't_start': float}
    """
    refs     = df["x_ref_cm"].values
    changes  = np.where(np.diff(refs) != 0)[0] + 1   # índices da mudança

    segments = []
    boundaries = list(changes) + [len(df)]

    # Primeiro segmento começa no índice 0
    starts = [0] + list(changes)

    for s, e in zip(starts, boundaries):
        if (e - s) < MIN_STEP_SAMPLES:
            continue
        ref_val  = refs[s]
        ref_prev = refs[s - 1] if s > 0 else np.nan
        segments.append({
            "ref":      ref_val,
            "ref_prev": ref_prev,
            "i_start":  s,
            "i_end":    e - 1,
            "t_start":  df["tempo_s"].iloc[s],
        })
    return segments


def settling_time(error_signal: np.ndarray, dt: float, tol: float) -> float | None:
    """
    Retorna o tempo (s) a partir do início do degrau em que o erro
    permanece dentro de ±tol·|degrau| para sempre.
    Retorna None se nunca acomodar.
    """
    n = len(error_signal)
    for i in range(n - 1, -1, -1):
        if abs(error_signal[i]) > tol:
            if i + 1 >= n:
                return None
            return (i + 1) * dt
    return 0.0


def metrics_for_segment(df: pd.DataFrame, seg: dict) -> dict:
    s, e   = seg["i_start"], seg["i_end"]
    chunk  = df.iloc[s:e + 1].copy()
    t0     = chunk["tempo_s"].iloc[0]
    t      = (chunk["tempo_s"] - t0).values
    pos    = chunk["posicao_cm"].values
    ref    = seg["ref"]
    x0     = pos[0]          # posição inicial do degrau
    delta  = ref - x0        # amplitude do degrau (pode ser negativa)

    if abs(delta) < 0.01:    # degrau desprezível
        return None

    # Estimativa de dt
    dt_arr = np.diff(t)
    dt     = float(np.median(dt_arr)) if len(dt_arr) > 0 else 0.05

    # Erro em relação ao setpoint
    error  = ref - pos

    # ── Tempo de subida (10% → 90%) ──────────────────────────────────
    lo = x0 + 0.10 * delta
    hi = x0 + 0.90 * delta
    if delta > 0:
        idx_lo = np.where(pos >= lo)[0]
        idx_hi = np.where(pos >= hi)[0]
    else:
        idx_lo = np.where(pos <= lo)[0]
        idx_hi = np.where(pos <= hi)[0]

    t_rise = float(t[idx_hi[0]] - t[idx_lo[0]]) if (len(idx_lo) and len(idx_hi)) else None

    # ── Overshoot / Undershoot ────────────────────────────────────────
    if delta > 0:
        peak     = float(pos.max())
        overshoot  = max(0.0, (peak - ref) / abs(delta) * 100)
        undershoot = 0.0
    else:
        peak     = float(pos.min())
        overshoot  = max(0.0, (ref - peak) / abs(delta) * 100)   # pico além do ref
        undershoot = 0.0

    # ── Tempo de acomodação ──────────────────────────────────────────
    ts_2 = settling_time(error, dt, TOL_2 * abs(delta) if abs(delta) > 0.5 else TOL_2)
    ts_5 = settling_time(error, dt, TOL_5 * abs(delta) if abs(delta) > 0.5 else TOL_5)

    # ── Erro em regime permanente (últimos 20% do segmento) ──────────
    n_steady   = max(10, int(0.20 * len(pos)))
    ess        = float(np.mean(error[-n_steady:]))
    pos_steady = float(np.mean(pos[-n_steady:]))

    return {
        "label":       f"{seg['ref_prev']:.0f}→{ref:.0f} cm",
        "ref":         ref,
        "ref_prev":    seg["ref_prev"],
        "delta":       delta,
        "x0":          x0,
        "peak":        peak,
        "overshoot_%": round(overshoot, 2),
        "undershoot_%":round(undershoot, 2),
        "t_rise_s":    round(t_rise, 3) if t_rise is not None else None,
        "ts_2pct_s":   round(ts_2,   3) if ts_2  is not None else None,
        "ts_5pct_s":   round(ts_5,   3) if ts_5  is not None else None,
        "ess_cm":      round(ess,     4),
        "pos_steady":  round(pos_steady, 4),
        "t_start":     seg["t_start"],
        "t":           t,
        "pos":         pos,
        "ref_arr":     np.full_like(pos, ref),
        "error":       error,
        "u":           chunk["u_K"].values,
    }


# ─────────────────────────────────────────────────────────────────────
# Plot principal
# ─────────────────────────────────────────────────────────────────────

def plot_overview(df: pd.DataFrame, results: list[dict], save_path: Path):
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(3, 1, hspace=0.12, height_ratios=[3, 1.2, 1.2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    t_all   = df["tempo_s"].values
    pos_all = df["posicao_cm"].values
    ref_all = df["x_ref_cm"].values
    u_all   = df["u_K"].values
    err_all = ref_all - pos_all

    ax1.plot(t_all, pos_all, color='#2166ac', lw=1.4, label=r'$x(t)$ — posição')
    ax1.plot(t_all, ref_all, color='#d73027', lw=1.1, ls='--', label=r'$x_{ref}$ — setpoint')
    ax1.set_ylabel('Posição (cm)')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_title("Ball and Beam — Visão Geral", fontweight='bold')

    # Sombreia cada segmento
    for i, r in enumerate(results):
        if r is None: continue
        t0 = r["t_start"]
        t1 = df["tempo_s"].iloc[min(
            df.index.get_loc(df[df["tempo_s"] >= t0].index[0]) + len(r["t"]),
            len(df) - 1
        )]
        ax1.axvspan(t0, t1, alpha=0.07, color=CORES[i % len(CORES)])

    ax2.plot(t_all, err_all, color='#8856a7', lw=1.2)
    ax2.axhline(0, color='k', lw=0.8, ls=':')
    ax2.set_ylabel('Erro (cm)')

    ax3.plot(t_all, u_all, color='#1a9641', lw=1.2)
    ax3.axhline(0, color='k', lw=0.8, ls=':')
    ax3.set_ylabel('u(t) (graus)')
    ax3.set_xlabel('Tempo (s)')

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"  Salvo: {save_path}")
    return fig


def plot_steps(results: list[dict], save_path: Path):
    valid = [r for r in results if r is not None]
    n     = len(valid)
    if n == 0:
        print("  Nenhum degrau válido para plotar individualmente.")
        return

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    # Altura extra por linha para acomodar suptitle + títulos sem sobreposição
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.2 * rows),
                             squeeze=False)
    fig.suptitle("Ball and Beam — Resposta por Degrau", fontweight='bold',
                 fontsize=13, y=0.98)

    for idx, r in enumerate(valid):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        cor = CORES[idx % len(CORES)]

        ax.plot(r["t"], r["pos"],     color=cor,      lw=1.5, label='posição')
        ax.plot(r["t"], r["ref_arr"], color='#d73027', lw=1.0, ls='--', label='setpoint')

        # Banda ±2%
        delta_abs = abs(r["delta"])
        bnd = TOL_2 * delta_abs if delta_abs > 0.5 else TOL_2
        ax.axhspan(r["ref"] - bnd, r["ref"] + bnd, alpha=0.12,
                   color='#d73027', label='±2 %')

        # Anotações
        ann_lines = []
        if r["t_rise_s"]   is not None: ann_lines.append(f"$t_r$ = {r['t_rise_s']:.2f} s")
        if r["ts_2pct_s"]  is not None: ann_lines.append(f"$t_s$(±2%) = {r['ts_2pct_s']:.2f} s")
        if r["ts_5pct_s"]  is not None: ann_lines.append(f"$t_s$(±5%) = {r['ts_5pct_s']:.2f} s")
        ann_lines.append(f"OS = {r['overshoot_%']:.1f} %")
        ann_lines.append(f"$e_{{ss}}$ = {r['ess_cm']:.3f} cm")

        ax.text(0.97, 0.05, "\n".join(ann_lines),
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

        # Título apenas com o rótulo do degrau — suptitle já dá o contexto geral
        ax.set_title(r["label"], fontweight='bold', pad=6)
        ax.set_xlabel('Tempo relativo (s)')
        ax.set_ylabel('Posição (cm)')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # Esconde eixos extras
    for idx in range(len(valid), rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    # rect deixa margem no topo para o suptitle não colidir com os títulos dos subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"  Salvo: {save_path}")
    return fig


def print_table(results: list[dict]):
    valid = [r for r in results if r is not None]
    if not valid:
        return

    header = (
        f"{'Degrau':<14} {'Δ (cm)':>8} {'t_rise (s)':>11} "
        f"{'ts ±2% (s)':>11} {'ts ±5% (s)':>11} "
        f"{'OS (%)':>8} {'ess (cm)':>10}"
    )
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for r in valid:
        tr  = f"{r['t_rise_s']:.2f}"  if r['t_rise_s']  is not None else "—"
        ts2 = f"{r['ts_2pct_s']:.2f}" if r['ts_2pct_s'] is not None else "—"
        ts5 = f"{r['ts_5pct_s']:.2f}" if r['ts_5pct_s'] is not None else "—"
        print(
            f"{r['label']:<14} {r['delta']:>8.1f} {tr:>11} "
            f"{ts2:>11} {ts5:>11} "
            f"{r['overshoot_%']:>8.1f} {r['ess_cm']:>10.4f}"
        )
    print("─" * len(header) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = find_csv()

    print(f"\n  Arquivo: {csv_path}")
    df = load_data(csv_path)
    print(f"  Amostras: {len(df)}  |  Duração: {df['tempo_s'].iloc[-1] - df['tempo_s'].iloc[0]:.1f} s")

    segments = detect_steps(df)
    print(f"  Degraus detectados: {len(segments)}")
    for s in segments:
        print(f"    {s['ref_prev']:.0f} → {s['ref']:.0f} cm  "
              f"(t={s['t_start']:.1f}s, {s['i_end']-s['i_start']+1} amostras)")

    results = [metrics_for_segment(df, s) for s in segments]
    print_table(results)

    out_dir = csv_path.parent
    stem    = csv_path.stem

    plot_overview(df, results, out_dir / f"{stem}_overview.png")
    plot_steps(results,        out_dir / f"{stem}_steps.png")

    plt.show()
    print("  Concluído.\n")


if __name__ == "__main__":
    main()