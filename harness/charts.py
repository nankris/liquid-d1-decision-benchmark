"""Publication charts derived exclusively from saved results; no network calls."""
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/d1-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from sklearn.metrics import confusion_matrix
from common import ROOT, RAW, read_jsonl, successful
from tasks import TASKS

IMAGES = ROOT / 'images'
MODELS = ['d1-local', 'qwen-local', 'tfidf-lr']
NAMES = {'d1-local': 'd1-3B (open weights, local)', 'qwen-local': 'Qwen2.5-3B-Instruct (chat, local)',
         'tfidf-lr': 'TF-IDF + logistic regression (trained)', 'qwen-local-names': 'Qwen2.5-3B, label-name prompt'}
PALETTE = dict(zip(MODELS, ['#5B3FD9', '#F28C28', '#9AA0A6']))
TASK_NAMES = {'spam': 'Spam', 'toxic': 'Toxicity', 'routing': 'Ticket routing', 'rating': 'Review stars'}
SUBTITLE = 'n=200 per task · 4-bit GGUF on 8 CPU threads · run 10 Oct 2026'
FOOTER = 'Source: own runs, logs in results/raw'


def save(fig, name, title, extra=None):
    fig.text(.055, .96, title, va='top', fontsize=15, weight='bold', linespacing=1.35)
    title_lines = title.count('\n') + 1
    fig.text(.055, .96 - .043 * title_lines, SUBTITLE + ('\n' + extra if extra else ''),
             va='top', fontsize=8, color='#777777', linespacing=1.5)
    fig.text(.055, .025, FOOTER, color='#888888', fontsize=7)
    fig.savefig(IMAGES / name, dpi=200)
    plt.close(fig)
    print(f'Wrote images/{name}')


def calibration5(records):
    p = np.array([r['probs']['1'] for r in records])
    y = np.array([str(r['label']) == '1' for r in records])
    index = np.minimum((p * 5).astype(int), 4)
    bins = [(p[index == b].mean(), y[index == b].mean(), int((index == b).sum()))
            for b in range(5) if (index == b).any()]
    # ECE includes every occupied bin; the plot suppresses bins with fewer than 5 items.
    return sum(abs(x - y) * n for x, y, n in bins) / len(p), [b for b in bins if b[2] >= 5]


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                        'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#E9E9EF',
                        'axes.axisbelow': True, 'figure.facecolor': 'white'})
    IMAGES.mkdir(exist_ok=True)
    stats = {(s['model'], s['task']): s for s in json.loads((ROOT / 'results/summary.json').read_text())}
    rows = {(m, t): successful(read_jsonl(RAW / f'{m}__{t}.jsonl')) for m in MODELS for t in TASK_NAMES}
    with (ROOT / 'results/cost_projection.csv').open() as f:
        costs = list(csv.DictReader(f))
    cost_models = list(dict.fromkeys(r['model'] for r in costs))
    means = {m: np.mean([float(r['usd_per_1000']) for r in costs if r['model'] == m]) for m in cost_models}
    d1_cost = means['d1 (hosted, list)']

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=.09, right=.97, top=.78, bottom=.28)
    for j, t in enumerate(TASK_NAMES):
        models = MODELS + (['qwen-local-names'] if t == 'routing' else [])
        for i, m in enumerate(models):
            if m == 'qwen-local-names':
                variant = successful(read_jsonl(RAW / 'qwen-local-names__routing.jsonl'))
                correct = np.array([str(r['pred']) == str(r['label']) for r in variant])
                acc = correct.mean()
                boot = correct[np.random.default_rng(0).integers(0, len(correct), (1000, len(correct)))].mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
            else:
                s = stats[m, t]
                acc, lo, hi = (s[k] for k in ['accuracy', 'accuracy_ci_low', 'accuracy_ci_high'])
            x = j + (i - (len(models) - 1) / 2) * .2
            ax.bar(x, acc, .18, color=PALETTE.get(m, '#F28C28'), alpha=.55 if m.endswith('names') else 1,
                   hatch='//' if m.endswith('names') else None, label=NAMES[m] if j == 0 or m.endswith('names') else None)
            ax.errorbar(x, acc, yerr=[[acc-lo], [hi-acc]], color='#454550', capsize=3, lw=1)
            ax.text(x, hi + .023, f'{acc:.1%}', ha='center', fontsize=8)
    # Count a 'clear win' only when the 95% CIs do not overlap.
    wins = sum(stats['d1-local', t]['accuracy_ci_low'] > stats['qwen-local', t]['accuracy_ci_high'] for t in TASK_NAMES)
    ax.set(xticks=range(4), xticklabels=TASK_NAMES.values(), ylim=(0, 1.16), ylabel='Accuracy')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, frameon=False, fontsize=8)
    save(fig, 'accuracy_by_task.png', f'd1 clearly beat the same-size chat model on {wins} of {len(TASK_NAMES)} tasks', 'Whiskers: 95% bootstrap CIs · "clearly" = non-overlapping CIs; toxicity was a statistical tie')

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.subplots_adjust(left=.08, right=.97, top=.78, bottom=.31, wspace=.28)
    for ax, t in zip(axes, ['spam', 'toxic']):
        ax.plot([0, 1], [0, 1], '--', color='#888888', label='Perfect calibration', lw=1)
        for m in MODELS:
            ece, bins = calibration5(rows[m, t])
            x, y, n = map(np.array, zip(*bins))
            ax.plot(x, y, color=PALETTE[m], lw=1, label=f'{NAMES[m]} · ECE {ece:.3f}')
            ax.scatter(x, y, s=n * 1.8, color=PALETTE[m], alpha=.75, edgecolors='white', linewidths=.5)
        ax.set(title=TASK_NAMES[t], xlabel='Mean predicted P(yes)', ylabel='Observed positive fraction', xlim=(-.04, 1.04), ylim=(-.04, 1.04))
        ax.legend(loc='upper left', bbox_to_anchor=(-.03, -.23), fontsize=6.8, frameon=False)
    save(fig, 'reliability.png', 'Probability calibration varies by model and task', 'Five equal-width bins · only bins with ≥5 items shown · marker area reflects count')

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=.18, right=.97, top=.78, bottom=.20)
    maximum = max(stats[m, t]['latency_p95_ms'] for m in MODELS[:2] for t in TASK_NAMES)
    for j, t in enumerate(TASK_NAMES):
        for i, m in enumerate(MODELS[:2]):
            y = j * 2.7 + i
            s = stats[m, t]
            p50, p95 = s['latency_p50_ms'], s['latency_p95_ms']
            ax.barh(y, p50, height=.65, color=PALETTE[m], label=NAMES[m] if j == 0 else None)
            ax.plot([p50, p95], [y, y], color=PALETTE[m], lw=1)
            ax.plot(p95, y, '|', color=PALETTE[m])
            ax.text(p95 + maximum * .025, y, f'p50 {p50:.0f} ms / p95 {p95:.0f} ms', va='center', fontsize=8)
    ax.set(yticks=np.arange(4)*2.7+.5, yticklabels=TASK_NAMES.values(), xlim=(0, maximum*1.48), xlabel='Latency (ms)')
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.45, -.17), frameon=False, ncol=2, fontsize=8)
    save(fig, 'latency.png', 'Local latency depends on the task and the model', 'TF-IDF baseline: <1 ms (not shown) · bars: p50; thin whiskers: p95')

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=.25, right=.96, top=.69, bottom=.17)
    cost_names = ['d1 (hosted, list)', 'GPT-6 Luna', 'Claude Haiku 5.5', 'Gemini 3.1 Flash Lite', 'GPT-6.1 Sol', 'Claude Opus 5.5']
    lower = min(means.values()) / 2
    for i, m in enumerate(cost_models):
        v = means[m]
        ax.barh(i, v-lower, left=lower, height=.57, color=PALETTE['d1-local'] if i == 0 else str(.44 + .065*i))
        ax.text(v*1.12, i, f'${v:.4f} · {v/d1_cost:.0f}x', va='center', fontsize=9)
    ax.set(xscale='log', xlim=(lower, max(means.values())*3.3), yticks=range(len(cost_models)), yticklabels=cost_names,
           xlabel='USD per 1,000 decisions · mean across four tasks')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:g}'))
    save(fig, 'cost_per_1k.png', 'Projected API cost per 1,000 decisions (list prices)',
         'd1 tokens logged from d1-3B; chat models use logged Qwen prompt tokens + 2 output tokens\nat OpenRouter list prices on 10 Oct 2026. Lower bound for chat (no reasoning tokens).')

    multi = list({(r['id'], r['k']): r for r in read_jsonl(RAW / 'd1-local__multiq.jsonl') if not r.get('warmup') and not r.get('error')}.values())
    ks = sorted({r['k'] for r in multi})
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.subplots_adjust(left=.09, right=.96, top=.72, bottom=.19, wspace=.3)
    for ax, field, label in zip(axes, ['latency_ms', 'input_tokens'], ['Median latency (ms)', 'Median input tokens']):
        values = [np.median([r[field] for r in multi if r['k'] == k]) for k in ks]
        ax.plot(ks, values, 'o-', color=PALETTE['d1-local'])
        for k, v in zip(ks, values):
            ax.annotate(f'{v:,.0f}', (k, v), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
        if field == 'latency_ms':
            single = values[ks.index(1)]
            ax.plot(ks, np.array(ks)*single, '--', color='#999999', label='k × single-question latency')
            ax.legend(loc='upper left', fontsize=7, frameon=False)
        ax.set(xticks=ks, xlabel='Questions in one request', ylabel=label, ylim=(0, max(ax.get_ylim()[1], max(values)*1.2)))
        ax.margins(x=.1)
    _med = {k: np.median([r['latency_ms'] for r in multi if r['k'] == k]) for k in sorted({r['k'] for r in multi})}
    _kmax = max(_med)
    save(fig, 'multiq.png', f'{_kmax} questions in one request took {_med[_kmax] / _med[1]:.1f}x the time of one, not {_kmax}x',
         f'Multi-question experiment: {len({r["id"] for r in multi})} routing messages x k in (1, 2, 3, 5) · local llama.cpp, CPU')

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    fig.subplots_adjust(left=.13, right=.96, top=.77, bottom=.29, wspace=.45)
    labels = TASKS['routing'].options
    for ax, m, cmap in zip(axes, MODELS[:2], ['Purples', 'Oranges']):
        records = rows[m, 'routing']
        cm = confusion_matrix([r['label'] for r in records], [r['pred'] for r in records], labels=labels, normalize='true')
        ax.imshow(cm, vmin=0, vmax=1, cmap=cmap)
        ax.grid(False)
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=[s.title() for s in labels], yticklabels=[s.title() for s in labels], xlabel='Predicted team', ylabel='True team')
        ax.set_title(NAMES[m] + ('\nLetter prompt' if m == 'qwen-local' else '') + f'\nAccuracy {stats[m, "routing"]["accuracy"]:.1%}', fontsize=9, pad=12)
        plt.setp(ax.get_xticklabels(), rotation=60, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, f'{cm[i,j]:.0%}', ha='center', va='center', color='white' if cm[i,j]>.5 else '#333333', fontsize=5.5)
    save(fig, 'confusion_routing.png', 'd1 separates support teams more accurately than the letter prompt', 'Row-normalised shares · both colour scales run from 0% to 100%')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    fig.subplots_adjust(left=.09, right=.96, top=.72, bottom=.17, wspace=.27)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes, MODELS[:2]):
        records = rows[m, 'rating']
        x = np.array([int(r['label'])+1 for r in records])
        y = np.array([r['expected_score'] for r in records])+1
        s = stats[m, 'rating']
        ax.scatter(x+rng.uniform(-.12,.12,len(x)), y, alpha=.4, s=16, color=PALETTE[m])
        ax.plot(range(1,6), [y[x==k].mean() for k in range(1,6)], 'o-', color=PALETTE[m])
        ax.plot([1,5], [1,5], '--', color='#AAAAAA')
        ax.set(title=f'{NAMES[m]}\nExpected-score MAE {s["expected_score_mae_stars"]:.2f} · Spearman {s["spearman"]:.2f}',
               xticks=range(1,6), yticks=range(1,6), xlabel='True stars', ylabel='Expected stars', xlim=(.7,5.3), ylim=(.8,5.2))
        ax.title.set_fontsize(9)
    save(fig, 'rating_scatter.png', 'd1’s expected review ratings track the true stars more closely')

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=.09, right=.96, top=.80, bottom=.17)
    for label, title, shade in [('0','Ham','#9AA0A6'),('1','Spam','#5B3FD9')]:
        ax.hist([r['probs']['1'] for r in rows['d1-local','spam'] if str(r['label'])==label], bins=np.linspace(0,1,21), color=shade, alpha=.7, label=title)
    ax.set(xlabel='d1 P(spam)', ylabel='Messages')
    ax.legend(frameon=False)
    save(fig, 'confidence_hist.png', "d1's spam probabilities are mostly near 0 or 1")

    fig = plt.figure(figsize=(8,4.5))
    fig.text(.055,.77,'d1-3B vs a\nsame-size\nchat model', fontsize=24, weight='bold', va='top', color='#292336', linespacing=1.2)
    ax = fig.add_axes([.46,.30,.51,.49])
    ax.axis('off')
    table = ax.table(cellText=[[TASK_NAMES[t]]+[f'{stats[m,t]["accuracy"]:.1%}' for m in MODELS] for t in TASK_NAMES],
                     colLabels=['Task','d1 acc','Qwen acc','TF-IDF acc'], colWidths=[.34,.21,.23,.22], bbox=[0,0,1,1], cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#EEEAFB' if c==1 else '#F3F4F6' if r%2 else '#FAFAFC')
        if r==0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E6E3ED')
    fig.text(.055,.21,f'd1 projected hosted cost: ${d1_cost:.4f} per 1,000 decisions', fontsize=12, weight='bold', color=PALETTE['d1-local'])
    fig.text(.055,.14,'Accuracy on held-out tasks · cost averaged across four tasks',fontsize=8,color='#777777')
    save(fig,'hero.png','')
    for obsolete in ['reliability_spam.png','reliability_toxic.png','routing_confusion_d1.png','routing_confusion_qwen.png']:
        (IMAGES / obsolete).unlink(missing_ok=True)

if __name__ == '__main__':
    main()
