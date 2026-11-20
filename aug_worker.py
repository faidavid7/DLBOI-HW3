"""Helper used only to speed up the CPU run: pre-trains the Section-5 runs (reverse order) in a second process.
Results land in the same results/ cache that the notebook reads; the notebook alone can reproduce everything."""
import re, matplotlib; matplotlib.use('Agg')
src = open('nb_src.py', encoding='utf-8').read()
exec(src.split('# %% [markdown]\n# ## 1. Data')[0])
exec(re.search(r'^AUGS = \{.*?^\}', src, flags=re.M | re.S).group(0))
for n, tf in reversed(list(AUGS.items())):
    if not (RES_DIR / f'{n}.json').exists():
        run(n, aug_tf=tf, arch='cnn', epochs=20, sched='cosine', aug=n)
