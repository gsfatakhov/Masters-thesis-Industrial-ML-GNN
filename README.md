# Masters-thesis-Industrial-ML-GNN

The code of experiments for the master's thesis: **Industrial ML: graph neural networks for fault diagnosis in multivariate sensor data**.

## Dataset (TEP)

The Tennessee Eastman Process (extended).

## Experiment 1 (Fault Diagnosis)

`Experiment_1.ipynb` - training of proposed GNN model with graph structure learning layer for Fault Diagnosis task.

## Experiment 2 (Quality of obtained adjacency matrices)

`Experiment_2.ipynb` - training of GNN model with another architecture without graph structure learning layer. Adjacency matrices were obtained in Experiment 1:

`corr_A.pt`

`direct_A.pt`

`relu_A.pt`

`uni_A.pt`

`und_A.pt`

Next, to compare adjacency matrix with TEP diagram, GNN model from Experiment 1 was trained with max number of edges for each node equal to 3.

## Experiment 3 (Model with several graph structure learning layers)

`Experiment_3.ipynb` - training of novel GNN model with several graph structure learning layers.
