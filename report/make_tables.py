"""results/*.json -> report/tables.tex (one LaTeX table body per experiment group) + report/numbers.tex macros."""
import json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / 'results'
out = Path(__file__).resolve().parent

def load(n):
    return json.loads((R / f'{n}.json').read_text())

def row(n, label=None):
    r = load(n); h = r['hist']; t = r['test']
    gap = h['train_acc'][-1] - h['val_acc'][-1]
    return (f"{label or n.replace('_', r'\_')} & {r['best_epoch']} & {h['train_acc'][-1]:.1f} & {r['val']['acc']:.1f} & "
            f"{f'{gap:+.1f}'.replace('-', '$-$')} & {r['val']['loss']:.3f} & {t['acc']:.1f} & {t['sens']:.1f} & {t['spec']:.1f} & {t['auc']:.1f} \\\\")

GROUPS = {
    'baseline': ['baseline_val16', 'baseline_s0', 'baseline_s1', 'baseline_s2'],
    'lr': ['lr_0.0001', 'lr_0.001', 'lr_0.01'],
    'bs': ['bs_16', 'bs_64', 'bs_256'],
    'opt': ['opt_sgd', 'opt_sgdm', 'opt_adam', 'opt_adamw'],
    'sched': ['sched_none', 'sched_step', 'sched_cosine', 'sched_plateau'],
    'epochs': ['ep_5', 'ep_10', 'sched_none', 'ep_60'],
    'wd': ['wd_0', 'wd_0.0001', 'wd_0.001', 'wd_0.01', 'wd_0.1'],
    'arch': ['mlp_d1_w64', 'mlp_d3_w64', 'mlp_d5_w64', 'mlp_d3_w256', 'mlp_d3_w64_nodrop', 'mlp_d3_w64_nobn',
             'cnn', 'cnn_nodrop', 'cnn_nobn', 'cnn_narrow'],
    'aug': ['aug_none', 'aug_hflip', 'aug_rot10', 'aug_affine', 'aug_crop', 'aug_bc', 'aug_medical',
            'aug_autoaug', 'aug_randaug', 'aug_trivial', 'aug_vflip_rot90'],
    'final': ['final_s0', 'final_s1', 'final_s2'],
    'summary': ['baseline_s0', 'lr_0.01', 'opt_sgd', 'sched_none', 'sched_step', 'wd_0.1', 'mlp_d3_w256', 'cnn', 'cnn_nobn',
                'aug_bc', 'aug_affine', 'aug_medical', 'aug_randaug', 'aug_vflip_rot90', 'final_s0', 'final_s1', 'final_s2'],
}
LABEL = {'sched_none': 'ep\\_30 (= sched\\_none)'}
with open(out / 'tables.tex', 'w', encoding='utf-8') as f:
    for g, names in GROUPS.items():
        names = [n for n in names if (R / f'{n}.json').exists()]
        f.write(f'\\def\\tab{g}{{%\n' + '\n'.join(row(n, LABEL.get(n) if g == 'epochs' else None) for n in names) + '}\n')

# macros for numbers quoted in the text
mac = {}
def ms(names, key, split='test'):
    v = np.array([load(n)[split][key] for n in names]); return f'{v.mean():.1f} $\\pm$ {v.std(ddof=1):.1f}'
B, F = GROUPS['baseline'][1:], GROUPS['final']
for k in ('acc', 'sens', 'spec', 'auc'):
    mac[f'base{k}'] = ms(B, k); mac[f'final{k}'] = ms(F, k)
mac['basevalacc'] = ms(B, 'acc', 'val'); mac['finalvalacc'] = ms(F, 'acc', 'val')
mac['finalcfg'] = load('final_s0')['cfg'].get('aug') or load('final_s0')['cfg']['arch']
with open(out / 'numbers.tex', 'w', encoding='utf-8') as f:
    for k, v in mac.items():
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n'.replace('_', '\\_') if k == 'finalcfg' else f'\\newcommand{{\\{k}}}{{{v}}}\n')
print('ok', list(mac.items()))
