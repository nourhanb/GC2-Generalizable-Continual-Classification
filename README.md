# [IEEE TMI 2024] GC2: Generalizable Continual Classification of Medical Images
Abstract— Deep learning models have achieved remarkable success in medical image classification. These models are typically trained once on the available annotated images and thus lack the ability of continually learning new tasks (i.e., new classes or data distributions) due to the problem of catastrophic forgetting. Recently, there has been more interest in designing continual learning methods to learn different tasks presented sequentially over time while preserving previously acquired knowledge. However, these methods focus mainly on preventing catastrophic forgetting and are tested under a closed-world assumption; i.e., assuming the test data is drawn from the same distribution as the training data. In this work, we advance the state-of-the-art in continual learning by proposing GC2 for medical image classification, which learns a sequence of tasks while simultaneously enhancing its out-of-distribution robustness. To alleviate forgetting, GC2 employs a gradual culpability-based network pruning to identify an optimal subnetwork for each task. To improve generalization, GC2 incorporates adversarial image augmentation and knowledge distillation approaches for learning generalized and robust representations for each subnetwork. Our extensive experiments on multiple benchmarks in a task-agnostic inference demonstrate that GC2 significantly outperforms baselines and other continual learning methods in reducing forgetting and enhancing generalization. 

<p align="center">
  <img src="Screenshot%202024-03-10%20231631.jpg" alt="alt text">
</p>


## Usage

### Setup
See the `requirements.txt` for environment configuration. 
```bash
pip install -r requirements.txt
```

### Datasets
We evaluate GC2 on three classification tasks: skin lesion classification from dermatoscopy images ([SKIN](#skin)), peripheral blood cell classification from microscopic images ([BLOOD](#blood)), and colon tissue classification from H&E stained histopathology images ([COLON](#blood)).

##### SKIN 
1. Download HAM10000 (HAM) dataset form [here](https://www.nature.com/articles/sdata2018161).

2. Download Dermofit (DMF) from [here](https://licensing.edinburgh-innovations.ed.ac.uk/i/software/dermofit-image-library.html).

3. Download Derm7pt (D7P) from [here](http://derm.cs.sfu.ca/).

4. Download MSK from [here](https://arxiv.org/abs/1710.05006).

5. Download UDA from [here](https://isic-archive.com/).

6. Download BCN from [here](https://challenge2019.isic-archive.com/data.html).

7. Download PH2 from [here](https://www.fc.up.pt/addi/ph2%20database.html).

Note: While most datasets are accessible online, access to certain datasets may require payment.

##### BLOOD
8. Download  PBS-HCB from [here](https://figshare.com/articles/figure/PBCI-DS_A_Benchmark_Peripheral_Blood_Cell_Image_Dataset_for_Object_Detection/24417049). 

##### COLON
9. Download  NCT-CRC-HE from [here](https://www.kaggle.com/datasets/imrankhan77/nct-crc-he-100k
). 


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
jupyter notebook samples.ipynb
```

### Citation 
If you use this code in your research, please consider citing:

```text
@inproceedings{bayasi2024gc2,
  title={{GC2}: Generalizable Continual Classification of Medical Images},
  author={Bayasi, Nourhan and Hamarneh, Ghassan and Garbi, Rafeef},
  booktitle={{IEEE Transactions on Medical Imaging (TMI)}},
  year={2024}}
```