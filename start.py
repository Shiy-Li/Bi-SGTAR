import torch
import pandas as pd
from parser import parameter_parser
from main import main
from de_novo import denovo
from utils import set_seed

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    args = parameter_parser()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    model_types = ['BiSGTAR']
    args.seed = 1

    for i in range(9):
        args.data = i + 1
        args.alpha = 0.5
        args.beta = 0.4
        args.gama = 0.05
        if args.data == 8:
            args.alpha, args.beta, args.gama = 0.8, 0.8, 0.8
            args.weight_decay = 1e-8
            args.epochs = 600
        elif args.data == 9:
            args.alpha, args.beta, args.gama = 0.8, 0.6, 0.8
            args.weight_decay = 1e-10
            args.epochs = 400

        for each in model_types:
            args.model_type = each
            print('Now model is: ', args.model_type)
            set_seed(args.seed)
            main(args)
            # denovo(args)
