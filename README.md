# GC2: Generalizable Continual Classification of Medical Images



If you use this code in your research, please consider citing:

```text
@inproceedings{bayasi2024gc2,
  title={{GC2}: Generalizable Continual Classification of Medical Images},
  author={Bayasi, Nourhan and Hamarneh, Ghassan and Garbi, Rafeef},
  booktitle={{IEEE Transactions on Medical Imaging (TMI)}},
  year={2024}}
```

## Usage

### Setup
See the `requirements.txt` for environment configuration. 
```bash
pip install -r requirements.txt
```

### Datasets


### Training

#### Continual Learning Phase
```bash
cd warmup/main/
python train.py
```

#### Domain Generalization Phase
```bash
cd main/main/
python train.py
```

#### OOD Visualization
```bash
cd main/main/
python samples.ipynb
```