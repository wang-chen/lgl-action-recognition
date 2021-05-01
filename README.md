# Lifelong Graph Learning

   This repo is for the application in paper "Bridging Graph Network to Lifelong
   Learning with Feature Correlation".

   **Temporal and distributed pattern recognition** using
   the Wearable Action Recognition Dataset (WARD).

# Training and Testing

   For feature graph network (FGN):

    python nonlifelong.py --model FGN --optim SGD
    python lifelong.py --model FGN --optim SGD

   For graph attention network (GAT):

    python nonlifelong.py --model GAT --optim Adam
    python lifelong.py --model FGN --optim Adam

   You can also specify the dataset location to be downloaded (Default: /data/datasets). For example:

    python nonlifelong.py --data-root ./ --model FGN --optim SGD

# Reproduce results in the paper

   Download [pre-trained models](https://github.com/wang-chen/graph-action-recognition/releases/download/v1.0/saves.zip). Then run:
   
    python evaluation.py --load saves/lifelong-FGN-s0.model
    python evaluation.py --load saves/lifelong-GAT-s0.model
    python evaluation.py --load saves/nonlifelong-FGN-s0.model
    python evaluation.py --load saves/nonlifelong-GAT-s0.model
   
   We provide all snapshot models during training, which is named as "[task]-[model]-s[seed]-it[iteration].model". 
   
   For example, "lifelong-FGN-s0-it3000.model"


# Citation

    @article{wang2020lifelong,
      title={Lifelong Graph Learning},
      author={Wang, Chen and Qiu, Yuheng, Gao, Dasong and Scherer, Sebastian},
      journal={arXiv preprint arXiv:2009.00647},
      year={2020}
    }
