import torch
import torch.nn as nn
import torch.nn.functional as F


def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]
    return name


def train(model, y0, args, alpha, i, rel):
    loss_list = []
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    for e in range(args.epochs):
        model.train()
        yl, rna_feat, rna_quality, hc, yd, dis_feat, dis_quality, hd = model(y0)
        y = alpha * yl + (1 - alpha) * yd.t()

        rna_confidence = torch.mul(hc, rel)
        dis_confidence = torch.mul(hd, rel.t())

        # Modified loss function
        # Intra-view loss
        # RNA
        rna_SPC = torch.mean(rna_quality)
        rna_TAR = criterion(hc, rel) + F.mse_loss(yl, rna_confidence)
        rna_loss = args.beta * rna_TAR + (1-args.beta) * rna_SPC
        # Disease
        dis_SPC = torch.mean(dis_quality)
        dis_TAR = criterion(hd, rel.t()) + F.mse_loss(yd, dis_confidence)
        dis_loss = args.beta * dis_TAR + (1 - args.beta) * dis_SPC
        # Inter-view loss
        loss_inter = alpha * rna_loss + (1-alpha) * dis_loss
        loss_cls = criterion(y, rel)
        loss = args.gama * loss_cls + (1 - args.gama) * loss_inter

        loss_list.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            yl, _, _, zc, yd, _, _, zd = model(y0)
        # if e % 50 == 0:
        #   print('Epoch %d | Lossp: %.4f' % (e, lossp.item()))
    model.eval()
    yli, rna_feat, quality_yli, hc, ydi, dis_feat, quality_ydi, hd = model(y0)
    y = alpha * yli + (1 - alpha) * ydi.t()
    return y
