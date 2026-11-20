# HW3: Hyperparameter Tuning — Chest X-ray (NORMAL vs PNEUMONIA)

115061534 江承瀚 · Deep Learning in Biomedical Optical Imaging (11510IPT 553000)

| File | Content |
|---|---|
| `115061534_江承瀚_HW3.ipynb` | Code with all outputs: data analysis, baseline, and the experiments of report Sections 2–6 |
| `115061534_江承瀚_HW3.pdf` | Analysis report |
| `results/*.json` | Training history and validation/test metrics of every run |
| `figs/` | Figures used in the report |
| `report/` | LaTeX source of the report, and `make_tables.py`, which builds the report tables from `results/` |
| `aug_worker.py`, `final_worker.py` | Helper scripts that ran some experiments in a second process to save time (the same code as the notebook) |

## 下載
Click the green `Code` button and choose `Download ZIP`, or run `git clone`.

## 解壓縮
Unzip the file. You get a `DLBOI-HW3/` folder.

## 選擇資料夾
Open `DLBOI-HW3/` as the working directory in Jupyter or VS Code. The notebook uses relative paths only.

## Reproduce

1. Put the course dataset in `chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}`.
2. Delete `results/` if you want to retrain everything. Otherwise the notebook reuses the cached results.
3. Run `jupyter nbconvert --to notebook --execute --inplace 115061534_江承瀚_HW3.ipynb`.

The notebook trains on CPU. Retraining all runs takes about 4 h on 16 cores.

## Main result

The test set comes from a different distribution than the training data. The main errors are NORMAL images classified as PNEUMONIA.

- Baseline (starter MLP): about 80 % test accuracy, AUC about 90.
- Final model: a small CNN with RandAugment, selected only on a patient-grouped validation split. Over 3 seeds it reached **85.0 ± 2.6 %** test accuracy and **AUC 95.8 ± 0.4**. The baseline MLP reached 80.5 ± 3.1 % and AUC 90.1 ± 1.5.
- Best single run: a CNN with chest-X-ray–appropriate augmentation (small affine transforms and brightness/contrast) reached 88.9 % test accuracy and AUC 96.3.

## License
Licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). The dataset is not included.
