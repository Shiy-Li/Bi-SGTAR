import torch
from main import main
from parser import parameter_parser
from utils import set_seed
import pandas as pd

if __name__ == "__main__":
    args = parameter_parser()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    set_seed(args.seed, args.cuda)
    model_types = ['AECDA']
    count = 0
    metric_each = pd.DataFrame(columns=['model','accuracy','recall','precision','F1','AUC','AUPR'])
    for i in range(7):
        args.data = i + 1
        if args.data == 6:
            args.epochs = 600
            args.alpha = 0.8
            args.beta = 0.8
            args.gama = 0.8
            args.weight_decay = 1e-8
        elif args.data == 7:
            args.epochs = 500
            args.alpha = 0.7
            args.beta = 0.5
            args.gama = 0.8
            args.weight_decay = 1e-10
        for each in model_types:
            args.model_type = each
            print('Now model is: ', args.model_type)
            # denovo(_,args,count)
            main(None, args, count)
            count += 1
    # metric_each.to_csv('./Experiments/Ablation_Bi-SGTAR.csv')
