import torch
import torch.nn as nn
import torch.nn.functional as F


class TAR(nn.Module):
    def __init__(self, hid_dim, out_dim, bias=False):
        super(TAR, self).__init__()
        # encoder-1
        self.e1 = nn.Linear(out_dim, hid_dim, bias=bias)
        # decoder
        self.d1 = nn.Linear(hid_dim, out_dim, bias=bias)

        self.Confidence = nn.Linear(hid_dim, out_dim, bias=bias)
        self.act1 = nn.ELU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.Sigmoid()

    def encoder(self, x):
        h = self.act1(self.e1(x))
        return h

    def decoder(self, z):
        h = self.act2(self.d1(z))
        return h

    def confidencer(self, z):
        y = self.act3(self.Confidence(z))
        return y

    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder(z)
        y = self.confidencer(z)
        return y, h


class BiSGTAR(nn.Module):
    def __init__(self, args):
        super(BiSGTAR, self).__init__()
        dis_num = args.dis_num
        rna_num = args.rna_num
        self.input_drop = nn.Dropout(0.)
        self.att_drop = nn.Dropout(0.)
        self.FeatQC_rna = nn.Linear(dis_num, dis_num, bias=True)
        self.FeatQC_dis = nn.Linear(rna_num, rna_num, bias=True)
        self.AE_rna = TAR(args.hidden, dis_num)
        self.AE_dis = TAR(args.hidden, rna_num)
        self.act = nn.Sigmoid()
        self.dropout = args.dropout

    def forward(self, feat):
        rna_quality = self.act(F.dropout(self.FeatQC_rna(feat), self.dropout))
        dis_quality = self.act(F.dropout(self.FeatQC_dis(feat.t()), self.dropout))

        rna_sparse_feat = torch.mul(rna_quality, feat)
        dis_sparse_feat = torch.mul(dis_quality, feat.t())

        yc, hc = self.AE_rna(rna_sparse_feat)
        yd, hd = self.AE_dis(dis_sparse_feat)

        return yc, rna_sparse_feat, rna_quality, hc, yd, dis_sparse_feat, dis_quality, hd