"""Convert nb_src.py ('# %%' cell markers) into the submission notebook."""
import re, sys, nbformat

src = open('nb_src.py', encoding='utf-8').read()
cells = []
for chunk in re.split(r'^# %%', src, flags=re.M)[1:]:
    head, _, body = chunk.partition('\n')
    if '[markdown]' in head:
        cells.append(nbformat.v4.new_markdown_cell(re.sub(r'^# ?', '', body.strip(), flags=re.M)))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nbformat.write(nb, sys.argv[1])
print(len(cells), 'cells ->', sys.argv[1])
