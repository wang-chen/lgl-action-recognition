# Lifelong Graph Learning

   This repo is for the application in paper "[Lifelong Graph Learning](https://arxiv.org/pdf/2009.00647.pdf)", CVPR, 2022.

   **Temporal and distributed pattern recognition** using the Wearable Action Recognition Dataset (WARD).

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

   Download [pre-trained models (v2.0)](https://github.com/wang-chen/graph-action-recognition/releases/download/v2.0/saves.zip) and extract. Then run:

    python evaluation.py --load saves/lifelong-fgn-s0.model
    python evaluation.py --load saves/lifelong-afgn-s0.model
    python evaluation.py --load saves/lifelong-appnp-s0.model
    python evaluation.py --load saves/lifelong-gcn-s0.model
    python evaluation.py --load saves/lifelong-gat-s0.model

   We provide all snapshot models during training, which is named as "[task]-[model]-s[seed]-it[iteration].model". 
   
   For example, "lifelong-fgn-s0-it3000.model"


# Citation

    @inproceedings{wang2022lifelong,
      title={Lifelong graph learning},
      author={Wang, Chen and Qiu, Yuheng and Gao, Dasong and Scherer, Sebastian},
      booktitle={2022 Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022}
    }
