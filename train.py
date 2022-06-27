import argparse
import torch
from models import *

try:
    from fddbenchmark import FDDDataset, FDDDataloader, FDDEvaluator
except:
    !git clone https://github.com/airi-industrial-ai/fddbenchmark
    from fddbenchmark import FDDDataset, FDDDataloader, FDDEvaluator


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tep_dataset = FDDDataset(name='reinartz_tep', splitting_type='supervised')
    tep_dataset.df = (tep_dataset.df - tep_dataset.df.mean()) / tep_dataset.df.std()
    tep_dataset.df['xmv_5'] = 1.0
    tep_dataset.df['xmv_9'] = 1.0
    train_dl = FDDDataloader(
        dataframe=small_tep.df,
        mask=small_tep.train_mask,
        labels=small_tep.labels,
        window_size=args.window_size,
        step_size=args.step,
        minibatch_training=True,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_dl = FDDDataloader(
        dataframe=small_tep.df,
        mask=small_tep.test_mask,
        labels=small_tep.labels,
        window_size=args.window_size,
        step_size=args.step,
        minibatch_training=True,
        batch_size=args.batch_size
    )

    if args.model == 'gnn':
        model = GNN(args.nnodes, args.window_size, args.ngnn, args.gsllayer, args.nhidden,
                    args.alpha, args.k, device=device)
    elif args.model == '1dcnn':
        model = CNN1DTEP(args.batch_size, args.window_size, args.nnodes)
    elif args.model == 'mlp':
        model = nn.Sequential(
                             nn.Linear(args.nnodes*args.window_size, 512), nn.ReLU(),
                             nn.Linear(512, 29),
                             )
    else:
        print('wrong model\'s name!')

    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    for epoch in range(args.num_epochs):



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gnn')
    parser.add_argument('--nnodes', type=int, default=52,
                        help='number of nodes')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--ngnn', type=int, default=1,
                        help='number of model trained in parallel')
    parser.add_argument('--gsllayer', type=str, default='directed',
                        help='type of graph structure layer')
    parser.add_argument('--nhidden', type=int, default=256,
                        help='number of parameters in gnn layer')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=50)
    args = parser.parse_args()
    main(args)
