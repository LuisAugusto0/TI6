#!/usr/bin/env python3
"""Plot workers vs execution time for weak scalability.

Each point's size represents `imagens_testadas` as a percentage of the largest
`imagens_testadas` in the CSV (so sizes are comparable across rows).

Usage:
    python plot_weak_scalability.py escalabilidade_fraca(x3threads).csv --out weak_plot.png

Dependencies:
    pip install pandas matplotlib seaborn
"""
from pathlib import Path
import argparse
import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot(csv_path: Path, out_path: Path = None, show: bool = False, ymax_seconds: float = None):
    df = pd.read_csv(csv_path)

    required = {'tempo_execucao_s', 'workers', 'imagens_testadas'}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f'CSV must contain columns: {required}')

    # Aggregate by workers (mean tempo, take imagens_testadas as max per worker)
    df_group = df.groupby('workers', as_index=False).agg(
        tempo_execucao_s=('tempo_execucao_s', 'mean'),
        imagens_testadas=('imagens_testadas', 'max')
    )

    # Compute percentage of imagens_testadas relative to maximum (for annotation)
    max_imgs = df_group['imagens_testadas'].max()
    df_group['imgs_pct'] = (df_group['imagens_testadas'] / max_imgs) * 100.0

    # Make workers a categorical axis so spacing is equal regardless of numeric gaps
    df_group['workers'] = df_group['workers'].astype(int)
    order = [str(w) for w in sorted(df_group['workers'].unique())]
    df_group['workers'] = df_group['workers'].astype(str)

    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 5))

    # Bar plot: categorical X (workers) evenly spaced, Y = tempo_execucao_s
    ax = sns.barplot(data=df_group, x='workers', y='tempo_execucao_s', order=order, ci=None, palette='Blues')

    # Determine baseline for annotation offset: use provided ymax_seconds if given
    ymax_ref = ymax_seconds if (ymax_seconds is not None) else df_group['tempo_execucao_s'].max()

    # Annotate each bar with absolute imagens and percentage on two lines
    for i, row in df_group.iterrows():
        x = i  # barplot positions are 0..n-1
        height = row['tempo_execucao_s']
        imgs = int(row['imagens_testadas'])
        pct = row['imgs_pct']
        offset = 0.02 * ymax_ref
        ax.text(x, height + offset, f"{imgs} imgs\n({pct:.0f}%)",
                ha='center', va='bottom', fontsize=9)

    ax.set_title('Escalabilidade fraca: Workers (categórico) vs Tempo de execução')
    ax.set_xlabel('Workers')
    ax.set_ylabel('Tempo de execução (s)')

    plt.tight_layout()

    # Apply optional Y-axis limit (upper bound) in seconds, if provided
    if ymax_seconds is not None:
        try:
            ax.set_ylim(0, float(ymax_seconds))
        except Exception:
            pass

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved plot to {out_path}')

    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser(description='Plot weak scalability workers vs tempo_execucao_s')
    p.add_argument('csv', type=Path, help='CSV file with tempo_execucao_s,workers,imagens_testadas,...')
    p.add_argument('--out', type=Path, default=Path('escalabilidade_fraca_plot.png'), help='Output image path')
    p.add_argument('--ymax-seconds', type=float, default=None, help='Optional: upper limit of Y axis in seconds')
    p.add_argument('--show', action='store_true', help='Show plot interactively')
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f'CSV not found: {args.csv}')

    plot(args.csv, args.out, args.show, ymax_seconds=args.ymax_seconds)


if __name__ == '__main__':
    main()
