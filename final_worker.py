"""Helper for the CPU run: trains final_s2, final_s1 in a second process (same code/config as notebook Section 6.1)."""
import re, json, sys, matplotlib; matplotlib.use('Agg')
src = open('nb_src.py', encoding='utf-8').read()
exec(src.split('# %% [markdown]\n# ## 1. Data')[0])
exec(re.search(r'^AUGS = \{.*?^\}', src, flags=re.M | re.S).group(0))
all_runs = sorted(p.stem for p in RES_DIR.glob('*.json') if not p.stem.startswith(('final', 'baseline_val16')))
summary = table(all_runs).sort_values(['val_acc', 'val_loss'], ascending=[False, True])
best_cfg = json.loads((RES_DIR / f'{summary.index[0]}.json').read_text())['cfg']
print('selected', summary.index[0], flush=True)
for s in ([int(a) for a in sys.argv[1:]] or (2, 1)):
    if not (RES_DIR / f'final_s{s}.json').exists():
        run(f'final_s{s}', aug_tf=AUGS.get(best_cfg['aug']), **{**best_cfg, 'epochs': 30, 'sched': 'cosine', 'seed': s})
