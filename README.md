# Lifelong Graph Learning

   This repo is for the application of paper "Bridging Graph Network to Lifelong
   Learning with Feature Correlation".

   
   Temporal and distributed pattern recognition using
   the Wearable Action Recognition Dataset (WARD).

   For feature graph network (FGN):

        python nonlifelong.py --model FGN --optim SGD --lr 1e-3
        python lifelong.py --memory-size 5000 --model FGN --optim SGD --lr 1e-3

   For graph attention network (GAT):

        python nonlifelong.py --model GAT --optim Adam --lr 1e-3
        python lifelong.py --memory-size 5000 --model GAT --optim Adam --lr 1e-3
