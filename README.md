# Semantic-guided Multimodal Feature Distillation in Recommendation
This is our Pytorch implementation for the paper

## Introduction
In this work,  In this paper, in order to yield useful multimodal information, we employ the knowledge distillation technique to distill valuable information in modality features from teacher model to student, and the teacher model captures the information in
the features considering both the semantic information from labels and the complementary information from multiple modalities.

## Environment Requirement
The code has been tested running under Python 3.5.2. The required packages are as follows:
- Pytorch == 1.7.0
- torch-cluster == 1.5.9
- torch-geometric == 1.6.3
- torch-scatter == 2.0.6
- torch-sparse == 0.6.9
- numpy == 1.19.2

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes.
- Clothing dataset  
```python -u main.py --l_r=0.0001 --weight_decay=0.0001 --dropout=0 --weight_mode=confid --num_routing=3 --is_pruning=False --data_path=Clothing```
- ToysGames dataset  
`python -u main.py --l_r=0.0001 --weight_decay=0.001 --dropout=0 --weight_mode=confid --num_routing=3 --is_pruning=False --data_path=ToysGames`
- Sports dataset  
`python -u main.py --l_r=0.0001 --weight_decay=0.01 --dropout=0 --weight_mode=confid --num_routing=3 --is_pruning=False --data_path=Sports`  

Some important arguments:  

- `weight_model` 
  It specifics the type of multimodal correlation integration. Here we provide three options:  
  1. `mean` implements the mean integration without confidence vectors. Usage `--weight_model 'mean'`
  2. `max` implements the max integration without confidence vectors. Usage `--weight_model 'max'`
  3. `confid` (by default)  implements the max integration with confidence vectors. Usage `--weight_model 'confid'`
  
- `fusion_mode` 
  It specifics the type of user and item representation in the prediction layer. Here we provide three options:  
  1. `concat` (by default) implements the concatenation of multimodal features. Usage `--fusion_mode 'concat'`
  2. `mean` implements the mean pooling of multimodal features. Usage `--fusion_mode 'max'`
  3. `id` implements the representation with only the id embeddings. Usage `--fusion_mode 'id'`
  

- `is_pruning` 
  It specifics the type of pruning operation. Here we provide three options:  
  1. `Ture` (by default) implements the hard pruning operations. Usage `--is_pruning 'True'`
  2. `False` implements the soft pruning operations. Usage `--is_pruning 'False'`
  
- 'has_v', 'has_a', and 'has_t' indicate the modality used in the model.

## Dataset
Please download the  [Amazon Review Data](https://nijianmo.github.io/amazon/index.html) as the datasets: Clothing, ToysGames, and Sports.

|#Interactions|#Users|#Items|#label|Visual|Textual|
|:-|:-|:-|:-|:-|:-|
|Clothing|18,209|17,318|26|2,048|1024|
|ToysGames|18,748|11,672|19|2048|1024|
|Sports|21,400|36,224|18|2,048|1024|

-`train.npy`
   Train file. Each line is a user with her/his positive interactions with items: (userID and micro-video ID)  
-`val.npy`
   Validation file. Each line is a user with her/his several positive interactions with items: (userID and micro-video ID)  
-`test.npy`
   Test file. Each line is a user with her/his several positive interactions with items: (userID and micro-video ID)  

