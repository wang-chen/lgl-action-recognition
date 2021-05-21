# Lifelong Graph Learning

   This repo is for the application in paper "Bridging Graph Network to Lifelong
   Learning with Feature Correlation".

   **Temporal and distributed pattern recognition** using
   the Wearable Action Recognition Dataset (WARD).

# Training and Testing

   **Note that MLP, AFGN and GAT perform the best with Adam, while the others perform the best with SGD.**

   For feature graph network (FGN):

    python regular.py --model FGN --optim SGD
    python lifelong.py --model FGN --optim SGD
    
   For attention feature graph network (AFGN):

    python regular.py --model AFGN --optim Adam
    python lifelong.py --model AFGN --optim Adam

   For multi-layer perceptron (MLP):

    python regular.py --model MLP --optim Adam
    python lifelong.py --model MLP --optim Adam

   For graph attention network (GAT):

    python regular.py --model GAT --optim Adam
    python lifelong.py --model GAT --optim Adam
  
   For grach convolutional network (GCN):

    python regular.py --model GCN --optim SGD
    python lifelong.py --model GCN --optim SGD
    
   For approximated personalized propagation of neural predictions (APPNP):

    python regular.py --model APPNP --optim SGD
    python lifelong.py --model APPNP --optim SGD

   You can also specify the dataset location to be downloaded (Default: /data/datasets). For example:

    python regular.py --data-root ./ --model FGN --optim SGD

# Reproduce results in the paper

   Download [pre-trained models](https://github.com/wang-chen/graph-action-recognition/releases/download/v1.0/saves.zip) and extract. Then run:
   
    python evaluation.py --load saves/lifelong-fgn-s0.model
    python evaluation.py --load saves/lifelong-afgn-s0.model
    python evaluation.py --load saves/lifelong-appnp-s0.model
    python evaluation.py --load saves/lifelong-gcn-s0.model
    python evaluation.py --load saves/lifelong-gat-s0.model

   We provide all snapshot models during training, which is named as "[task]-[model]-s[seed]-it[iteration].model". 
   
   For example, "lifelong-fgn-s0-it3000.model"


# Citation

    @article{wang2020lifelong,
      title={Lifelong Graph Learning},
      author={Wang, Chen and Qiu, Yuheng and Scherer, Sebastian},
      journal={arXiv preprint arXiv:2009.00647},
      year={2020}
    }
