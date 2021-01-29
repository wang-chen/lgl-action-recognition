# Lifelong Graph Learning

   This repo is for the application in paper "Bridging Graph Network to Lifelong
   Learning with Feature Correlation".

   Temporal and distributed pattern recognition using
   the Wearable Action Recognition Dataset (WARD).

   For feature graph network (FGN):

    python nonlifelong.py --model FGN --optim SGD
    python lifelong.py --model FGN --optim SGD

   For graph attention network (GAT):

    python nonlifelong.py --model GAT --optim Adam
    python lifelong.py --model FGN --optim Adam

   You can also specify the dataset location to be downloaded (Default: /data/datasets). For example:

    python nonlifelong.py --data-root ./ --model FGN --optim SGD
