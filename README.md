# Adopting and improving LSTMs as language models

## How to run

### Train

```bash
python3 run.py -c configs/baseline.yaml -t
```

### Evaluate

```bash
python3 run.py -c configs/baseline.yaml -e -d
```

### Inference

```bash
python3 run.py -c configs/baseline.yaml -i -p "the director"
```

## Configuration