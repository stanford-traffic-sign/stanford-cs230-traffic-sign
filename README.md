# Stanford CS230 Traffic Sign Group Project

## Paper

https://cs230.stanford.edu/projects_fall_2020/reports/55824835.pdf

## How to Run

### Prepare

Download the dataset at https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

Unzip it and move to GTSRB folder.

### Commands

Build dataset

```bash
python build_dataset.py
```

Train

```bash
python train.py
```

Evaluate

```bash
python evaluate.py
```

Summarize model

```bash
python summarize_model.py
```

PR Curves

```bash
python pr_curves.py
```

Visualize model

```bash
python visualize_model.py
```

Run TensorBoard

```bash
tensorboard --logdir=runs
```
