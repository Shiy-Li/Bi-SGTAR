import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--model-type', type=str, default='BiSGTAR',
                        help='choose the model.')
    parser.add_argument('--rna-num', type=int, default=0, help='circrna number.')
    parser.add_argument('--dis-num', type=int, default=0, help='disease number.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Dimension of representations')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight between lncRNA space and disease space')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='Weight between lncRNA space and disease space')
    parser.add_argument('--gama', type=float, default=0.05,
                        help='Weight between lncRNA space and disease space')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Weight between lncRNA space and disease space')

    parser.add_argument('--data', type=int, default=5,
                        help='Dataset')
    parser.add_argument('--para', type=float, default=1e-2,
                        help='Smooth Factor')

    args = parser.parse_known_args()[0]
    return args
