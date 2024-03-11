# [IEEE TMI 2024] GC2: Generalizable Continual Classification of Medical Images
Abstract— Deep learning models have achieved remarkable success in medical image classification. These models are typically trained once on the available annotated images and thus lack the ability of continually learning new tasks (i.e., new classes or data distributions) due to the problem of catastrophic forgetting. Recently, there has been more interest in designing continual learning methods to learn different tasks presented sequentially over time while preserving previously acquired knowledge. However, these methods focus mainly on preventing catastrophic forgetting and are tested under a closed-world assumption; i.e., assuming the test data is drawn from the same distribution as the training data. In this work, we advance the state-of-the-art in continual learning by proposing GC2 for medical image classification, which learns a sequence of tasks while simultaneously enhancing its out-of-distribution robustness. To alleviate forgetting, GC2 employs a gradual culpability-based network pruning to identify an optimal subnetwork for each task. To improve generalization, GC2 incorporates adversarial image augmentation and knowledge distillation approaches for learning generalized and robust representations for each subnetwork. Our extensive experiments on multiple benchmarks in a task-agnostic inference demonstrate that GC2 significantly outperforms baselines and other continual learning methods in reducing forgetting and enhancing generalization. 

<center>
![alt text](Screenshot%202024-03-10%20231631.jpg)
</center>

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