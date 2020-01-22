# RLFAT
Code and models for **Robust Local Features for Improving the Generalization of Adversarial Training (RLFAT)** accepted by ICLR 2020.

Please refer to https://drive.google.com/drive/folders/183Sb5q_RQbzeZkw-uQbpBd7S1yTnn-Au?usp=sharing

#### REQUIREMENTS

The code was tested with Python 3.6.5, Tensorflow 1.12.0, Keras 2.12, Numpy 1.15.4 and cv2 3.4.2

#### EXPERIMENTS

Code refer heavily to:  [PGD Adversarial Training](https://github.com/MadryLab/cifar10_challenge) 

The code consists of seven Python scripts and the file `config.json` that contains various parameter settings.

##### Running the code

- `python train_pgdat_RLFAT.py`:  train the model for $\mathrm{RLFAT}_{\mathrm{P}}$ , storing checkpoints along the way.
- `python train_trades_RLFAT.py`:  train the model for $\mathrm{RLFAT}_{\mathrm{T}}$ , storing checkpoints along the way.
- `python pgd_attack_test.py`:  applies the attack (PGD or CW) to the testing set and stores the resulting adversarial testing set in a `.npy` file. 
- `python run_attack_test.py`: evaluates the model on the testing examples in the `.npy` file specified in config.
- `python nattack.py`:  applies the Nattack to the testing set and report the attack failed number.

##### Parameters in `config.json`

We refer the readers to [PGD Adversarial Training](https://github.com/MadryLab/cifar10_challenge)  on more details about the parameters in `config.json`.

##### Example usage

After cloning the repository you can either train a new network or evaluate/attack one of our pre-trained networks.

- Training a new model for $\mathrm{RLFAT}_{\mathrm{T}}$ 

```
python train_pgdat_RLFAT.py
```

- Test the model under PGD or CW attack：

```
python pgd_attack_test.py
python run_attack_test.py
```

- Test the model under Nattack：

```
python nattack.py
```

#### Acknowledgments

Code refer heavily to:  [PGD Adversarial Training](https://github.com/MadryLab/cifar10_challenge) 
