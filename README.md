## Packages
The following packages were used. We recommend installing them to a `conda` environment.

- `matplotlib == 3.4.2`
- `numpy == 1.20.2`
- `python == 3.9.4`
- `pytorch == 1.8.0`
- `scikit-learn == 0.24.2`
- `scipy == 1.6.3`
- `torchvision == 0.9.0`
- `tqdm == 4.60.0`

## Figure 1

1. `cd Figure-1/`
2. `python main.py`

## Binary Classification

### To Plot

1. `cd Binary-Classification/`
2. `python plot.py`

### To Redo Experiments

1. `cd Binary-Classification/`
2. Compute and save results:
```
python main.py --data_decay 1
python main.py --data_decay 1e-1
python main.py --data_decay 1e-2
python main.py --data_decay 1e-3
python main.py --data_decay 1e-4
python main.py --data_decay 1e-5
python main.py --data_decay 1e-6
python main.py --data_decay 1e-8
python main.py --data_decay 1e-10
python main.py --data_decay 1e-12
python main.py --data_decay 1e-14
python main.py --data_decay 1e-16
python main.py --weight_decay 1
python main.py --weight_decay 1e-1
python main.py --weight_decay 1e-2
python main.py --weight_decay 1e-3
python main.py --weight_decay 1e-4
python main.py --weight_decay 1e-5
python main.py --weight_decay 1e-6
```

## Auto-Encoder

### To Plot

1. `cd Auto-Encoder/`
2. `Figure-3.py`, `Figure-4.py`, `Figure-5.py`, `Figure-6.py`, `Figure-7.py`

### To Redo Experiments

1. `cd Auto-Encoder/`
2. For a particular experiment, compute and save results: `python main.py`
3. Process these results by modifying and using the corresponding function in `process.py`.
4. Repeat (1) and (2) for all 5 experiments.