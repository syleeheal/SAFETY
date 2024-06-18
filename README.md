# SAFETY

This is the code repository of the paper, "You're Not Alone in Battle: Combat Threat Analysis Using Attention Networks and a New Open Benchmark," published in CIKM 2023.

Dataset, benchmark, and training details are provided in the Appendix.pdf.

This repository contains the proposed (1) **combat simulation dataset**, (2) **model SAFETY**, and (3) **online Appendix**.

**Note: In coming weeks, we will make more details about the dataset and codes available.**

---

# Basics

./train folder has codes to run all deep learning models.

./dataset folder has dataset zip files.

./run folder has codes to reproduce the experimental outcomes in the paper.

./best_hyperparam folder has txt files with best hyperparameters for each deep learning model.

statistics_eval.ipynb file has codes to check raw data statistics.

---

# Dataset

The raw datasets are the **1.zip**, **2.zip**, **3.zip**, **4.zip** files in the dataset folder, where each number indicates the assigned tactic.

Preprocessed data for training and evaluation is **data_dict** file.

## Step 1: Unzip the Raw Data

### Code

```bash
cd ./SAFETY/
unzip ./dataset/1.zip ./dataset/2.zip ./dataset/3.zip ./dataset/4.zip
```

This will return four folders within ./dataset. Each folder has four csv files, each with features and labels.

## Step 2: Preprocess

Run codes in table2dict.ipynb to generate data_dict.pkl. (*Note that the zip file of data_dict is already provided in ./dataset.*)

## Step 3: Check Statistics (Optional)

Run codes in statistics_eval.ipynb to check unzipped raw data statistics.

---

# Codes and Hyperparameters

To run the model code, refer to ./run/run.sh file.

### Example: Train Model

```bash
python ./train/main.py --pred y_int --stress-type none --model SAFETY # train to predict y_int
python ./train/main.py --pred y_atk --stress-type none --model SAFETY # train to predict y_atk
```

Running the code will automatically load the tuned best hyperparameters from ./best_hyperparam.
If the model does not have tuned hyperparameters in ./best_hyperparam, grid search the model's best hyperparameters

### Example: Hyperparameter Search

```bash
python ./train/main.py --pred y_int --stress-type none --model SAFETY --optimize hyperparam # search best hyperparameters for y_int prediction
python ./train/main.py --pred y_atk --stress-type none --model SAFETY --optimize hyperparam # search best hyperparameters for y_atk prediction
```

As described in the Appendix.pdf, the maximum observed time (t_max) is 300. Only 20 randomly sampled timestamps (t_sample_size) serve as model input.
To change the setting, modify --t-max and --t-sample-size in the command.
