#!/usr/bin/env python3
"""Plot batch_size vs execution time colored by number of workers.

Usage:
    python plot_batch_vs_time.py path/to/escalabilidade_forte.csv --out plot.png

Dependencies:
    pip install pandas seaborn matplotlib
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(csv_path: Path, out_path: Path = None, show: bool = False):
    df = pd.read_csv(csv_path)

    # Expecting columns: tempo_execucao_s,workers,batch_size,imagens_testadas
    if 'tempo_execucao_s' not in df.columns:
        raise SystemExit('CSV must contain a `tempo_execucao_s` column')

    # Treat workers as categorical for coloring
    df['workers'] = df['workers'].astype(str)
    df['batch_size'] = df['batch_size'].astype(int)

    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 5))

    # Scatter plot: x=batch_size, y=tempo_execucao_s, hue=workers
    ax = sns.scatterplot(
        data=df,
        x='batch_size',
        y='tempo_execucao_s',
        hue='workers',
        palette='tab10',
        s=80,
        alpha=0.9,
        edgecolor='w'
    )

    # Also draw a small jittered swarm-like connector per workers to show distribution
    try:
        sns.lineplot(
            data=df.groupby(['batch_size', 'workers'])['tempo_execucao_s'].mean().reset_index(),
            x='batch_size', y='tempo_execucao_s', hue='workers', palette='tab10',
            legend=False, linewidth=1, alpha=0.6, ax=ax
        )
    except Exception:
        pass

    ax.set_title('Batch size vs Tempo de execução (cores por workers)')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Tempo de execução (s)')

    # Improve legend title
    leg = ax.legend(title='workers')
    leg.set_title('workers')

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f'Saved plot to {out_path}')

    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser(description='Plot batch_size vs tempo_execucao_s colored by workers')
    p.add_argument('csv', help='CSV file (tempo_execucao_s,workers,batch_size,...)', type=Path)
    p.add_argument('--out', help='Output image path (png/svg)', type=Path, default=Path('batch_vs_time.png'))
    p.add_argument('--show', help='Show plot interactively', action='store_true')
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f'CSV file not found: {args.csv}')

    plot(args.csv, args.out, args.show)


if __name__ == '__main__':
    main()
