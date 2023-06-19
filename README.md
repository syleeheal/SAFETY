# SAFETY

The official code repository of the paper, "You're Not Alone in Battle: Combat Threat Analysis Using Attention Networks and a New Open Benchmark."

## Dataset
The proposed datset will automatically download at **dataset foler** when running main.py. 
Details with respect to dataset and benchmark task can be found in the **Appendix.pdf**.

## Usage
The shell files to run the codes are in the **run foler**.
For example, to reproduce results in Table 3 & 4 for SAFETY, use the following command:

**./run/run-sta.sh**

## Dependencies
numpy==1.21.2 

scikit_learn==1.2.2 

torch==1.11.0+cu113

torch_geometric==2.1.0

torch_scatter==2.0.9

torchmetrics==0.11.4

tqdm==4.62.3

xgboost==1.7.5

gdwon==4.7.1



