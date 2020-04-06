# EmpTransfo: A Multi-head Transformer Architecture for Creating Empathetic Dialog Systems

The present repo contains the code for the paper https://arxiv.org/abs/2003.02958
on empathetic dialog system. The repository is heavily influenced by https://github.com/huggingface/transfer-learning-conv-ai


## Installation
To install and use the training and inference scripts please clone the repo and install the requirements:

```bash
git clone git@github.com:roholazandie/EmpTransfo.git
cd EmpTransfo
pip install -r requirements.txt

```


## Interact with the chatbot
You can download the the checkpoint model [here](https://drive.google.com/open?id=1EjpK0YEVG1i9meLJzt7ZgODr0k65lTDi), extract and point to it from interact_config.json "model_checkpoint" value.
For example:
```
"model_checkpoint" : "/home/rohola/codes/EmpTransfo/emp_transfo_checkpoint"
``` 
Then run interact.py
```python
python interact.py
```

## Dataset
The original daily dialog dataset is [here](https://www.aclweb.org/anthology/I17-1099/). We changed the format to our purpose and can be download
from [here](https://drive.google.com/open?id=1T4AdY7wku8srL_xWSxgt-OHqdLFVo3s3). 


## Training

The script [train_multihead.py](https://github.com/roholazandie/EmpTransfo/blob/master/train_multihead.py) uses three heads with all features. 


The script [train_full.py](https://github.com/roholazandie/EmpTransfo/blob/master/train_full.py) uses two heads (next sentence prediction and LM head), but uses all the features.


The script [train_emotion_recognition.py](https://github.com/roholazandie/EmpTransfo/blob/master/train_emotion_recognition.py) trains to predict the next emotion (wihtout no_emotion).

The script [train.py](https://github.com/roholazandie/EmpTransfo/blob/master/train.py) trains without any features of the dataset (the base model).

For all training scripts just change the dataset_path in config.json file related to that task, and then run the script
without any arguments.



## Citation
If you use this code in your research, you can cite our ANLP paper:

```
@article{zandie2020emptransfo,
  title={EmpTransfo: A Multi-head Transformer Architecture for Creating Empathetic Dialog Systems},
  author={Zandie, Rohola and Mahoor, Mohammad H},
  journal={arXiv preprint arXiv:2003.02958},
  year={2020}
}
```