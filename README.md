# imdb_spoiler

INMAS Workshop 4 Project

## Configure Python environment

Suppose we have `python >= 3.12.x` installed on local machine.

1. Create a virtual environment for this project, in my laptop, I name it `env_torch` but you can choose any name you like. I put it in the folder `~/.venvs`, which is created by me

   ```bash
   python3 -m venv ~/.venvs/env_torch
   ```

2. Activate the `env_torch` virtual environment

   ```bash
   source ~/.venvs/env_torch/bin/acivate
   ```

3. Install all python modules needed

   ```bash
   pip install torch torchvision torchaudio
   pip install nltk
   pip install 'kagglehub[pandas-datasets]' # the single quotes are important because Zsh treats [] as special characters
   pip install ipykernel

   pip install tabulate

   pip install matplotlib

   pip install seaborn

   pip install -U scikit-learn

   pip install imbalanced-learn

   pip install transformers
   ```

4. Set Jupyter Kernel in VS Code. Open VSCode and press `Cmd + Shft + P`, type `python` and choose `select python interpreter` you'll see the interface below. If you don't see it, be patient and press the refresh buttuon and wait. It'll finally appear.

![1739592348320](image/README/1739592348320.png)

If it still doesn't show up, open terminal and activate `env_torch`, then type

```bash
python -m ipykernel install --user --name=env_torch --display-name "Python (env_torch)"
```

Now in the upper right corner of VSCode, you can see `Select Kernel` option. Click and choose `Python Environments`, and select `env_torch(Python 3.13.2)`

![1739592768552](image/README/1739592768552.png)

## Load JSON Data File

1. Download the  [IMDB Spoiler Dataset](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset). Click "Download" and choose the option "Download dataset as zip (348 MB)".
2. Unzip the .zip file, and ceate a folder named `data` in the project folder. Move the file `IMDB_reviews.json` (952.6 MB) into the folder data. And set the value of the variable `file_path` as `"data/IMDB_reviews.json"`.

## Downstream Tasks

### Prediction of Spoiler Labels

#### LSTM and BERT

Predict if there are spoilers in the movie review, which is a binary classification task. In this task, we trained 2 language models, LSTM and BERT to perform the binary classification. The LSTM is trained from scratch, and BERT is a pretrained model. The script for LSTM training is `lstm_train.py`. When you use it, you need to follow the command below, where '--batch_size', '--max_len' and '--n_epochs' are desired values for batch size of the dataloaders, length of token embedding and number of training epochs. Also `/path/to/json/file` is the location of the data (you can try `data/sampled_preprocessed.json`).

```bash
python lstm_train.py /path/to/json/file --batch_size 32 --max_len 256 --n_epochs 8

```


To train/finetune BERT, use the following similar command

```bash
python bert_train.py /path/to/json/file --batch_size 32 --max_len 256 --n_epochs 8
```

#### Distilled BERT (Sunny)


#### Long Transformers (Siqi)



### Howard and Yutong's Task 

To be added 