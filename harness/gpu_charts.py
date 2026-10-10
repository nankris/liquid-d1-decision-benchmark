"""GPU (Colab T4) charts from results/gpu_t4/d1_gpu_speed_results.json, in the style of charts.py.
CPU p50s for contrast come from results/summary.json. No network calls."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/d1-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT
from charts import PALETTE, TASK_NAMES

IMAGES = ROOT / 'images'
GPU_JSON = ROOT / 'results/gpu_t4/d1_gpu_speed_results.json'
FOOTER = 'Source: own runs, results/gpu_t4/d1_gpu_speed_results.json and results/summary.json'
GPU_TO_CPU = {'d1-3B': 'd1-local', 'qwen2.5-3b-instruct': 'qwen-local'}
TASKS = ['spam', 'routing']
GPU_NAMES = {'d1-3B': 'd1-3B', 'qwen2.5-3b-instruct': 'Qwen2.5-3B-Instruct (chat)'}


def save(fig, name, title, subtitle, extra=None, footer=FOOTER):
    fig.text(.055, .96, title, va='top', fontsize=15, weight='bold', linespacing=1.35)
    title_lines = title.count('\n') + 1
    fig.text(.055, .965 - .058 * title_lines, subtitle + ('\n' + extra if extra else ''),
             va='top', fontsize=8, color='#777777', linespacing=1.5)
    fig.text(.055, .025, footer, color='#888888', fontsize=7)
    fig.savefig(IMAGES / name, dpi=200)
    plt.close(fig)
    print(f'Wrote images/{name}')


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                        'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#E9E9EF',
                        'axes.axisbelow': True, 'figure.facecolor': 'white'})
    gpu = json.loads(GPU_JSON.read_text())
    cpu = {(s['model'], s['task']): s for s in json.loads((ROOT / 'results/summary.json').read_text())}
    pooled = {(r['model'], r['task']): r for r in gpu['pooled'] if r['parallel'] == 1}
    n_per_bar = {pooled[k]['n'] for k in pooled}
    gpu_name = gpu['gpu']['name'].replace('Tesla ', '')

    # Latency: GPU p50 bars with p95 whiskers, CPU p50 as faint bars behind.
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=.16, right=.97, top=.72, bottom=.20)
    maximum = max(cpu[GPU_TO_CPU[m], t]['latency_p50_ms'] for m in GPU_TO_CPU for t in TASKS)
    ratios = []
    for j, t in enumerate(TASKS):
        for i, m in enumerate(GPU_TO_CPU):
            y = j * 2.7 + i
            r = pooled[m, t]
            p50, p95 = r['latency_p50_ms'], r['latency_p95_ms']
            cpu_p50 = cpu[GPU_TO_CPU[m], t]['latency_p50_ms']
            colour = PALETTE[GPU_TO_CPU[m]]
            ax.barh(y, cpu_p50, height=.65, color=colour, alpha=.13,
                    label='CPU p50 (8 threads, earlier run)' if j == 0 and i == 0 else None)
            ax.text(cpu_p50 + maximum * .012, y, f'CPU {cpu_p50:.0f} ms', va='center', fontsize=7, color='#999999')
            ax.barh(y, p50, height=.65, color=colour, label=GPU_NAMES[m] + f' on {gpu_name} GPU' if j == 0 else None)
            ax.plot([p50, p95], [y, y], color=colour, lw=1)
            ax.plot(p95, y, '|', color=colour)
            ax.text(p95 + maximum * .012, y, f'p50 {p50:.0f} / p95 {p95:.0f} ms', va='center', fontsize=8)
        ratios.append(pooled['qwen2.5-3b-instruct', t]['latency_p50_ms'] / pooled['d1-3B', t]['latency_p50_ms'])
    ax.set(yticks=np.arange(len(TASKS)) * 2.7 + .5, yticklabels=[TASK_NAMES[t] for t in TASKS],
           xlim=(0, maximum * 1.18), xlabel='Latency per request (ms)')
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.45, -.13), frameon=False, ncol=3, fontsize=8)
    lo, hi = min(ratios), max(ratios)
    save(fig, 'gpu_latency.png',
         f'On a {gpu_name} GPU, d1 answered {lo:.1f}-{hi:.1f}x faster\nthan the same-size chat model',
         f'Colab {gpu_name} · llama.cpp b11541, all layers on GPU · 4-bit GGUF · one request at a time · '
         f'n={"/".join(map(str, sorted(n_per_bar)))} per bar (2 reps) · run 10 Oct 2026',
         'Bars: GPU p50; thin whiskers: GPU p95 · faint bars: CPU p50 from the 8-thread CPU run (n=200) · '
         'end-to-end HTTP latency, not architecture efficiency')

    # Multi-question: decisions/s and p50 vs k.
    rows = sorted(gpu['multiq']['rows'], key=lambda r: r['k'])
    ks = [r['k'] for r in rows]
    dps = [r['throughput_decisions_per_s'] for r in rows]
    p50 = [r['latency_p50_ms'] for r in rows]
    colour = PALETTE['d1-local']
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.subplots_adjust(left=.09, right=.96, top=.70, bottom=.19, wspace=.3)
    for ax, values, label, fmt in [(axes[0], dps, 'Decisions per second', '{:.1f}'),
                                   (axes[1], p50, 'Median latency per request (ms)', '{:,.0f}')]:
        ax.plot(ks, values, 'o-', color=colour)
        for k, v in zip(ks, values):
            ax.annotate(fmt.format(v), (k, v), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
        if values is p50:
            ax.plot(ks, np.array(ks) * p50[0], '--', color='#999999', label='k × single-question latency')
            ax.legend(loc='upper left', fontsize=7, frameon=False)
        ax.set(xticks=ks, xlabel='Questions in one request (k)', ylabel=label,
               ylim=(0, max(ax.get_ylim()[1], max(values) * 1.2)))
        ax.margins(x=.1)
    kmax = ks[-1]
    save(fig, 'gpu_multiq.png',
         f'On GPU, {kmax} questions in one request took {p50[-1] / p50[0]:.1f}x the time of one,\n'
         f'so decisions/s rose {dps[-1] / dps[0]:.1f}x ({dps[0]:.1f} to {dps[-1]:.1f})',
         f'd1-3B on a Colab {gpu_name} · {gpu["multiq"]["n_items"]} routing messages × k in (1, 2, 3, 5) · '
         'one request at a time · run 10 Oct 2026',
         'Decisions/s = requests/s × k, from real wall-clock time per batch',
         footer='Source: own runs, results/gpu_t4/d1_gpu_speed_results.json')


if __name__ == '__main__':
    main()
