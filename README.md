# Stanford CS230 Traffic Sign Group Project

## Prepare

Download the dataset at https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

Unzip it and move to GTSRB folder.

## Run

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

Visualize model

```bash
python summarize_model.py
```

TensorBoard

```bash
tensorboard --logdir=runs
```
