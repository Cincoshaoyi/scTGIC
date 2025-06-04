import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Module, Parameter,Dropout
import numpy as np
import opt


class AE_encoder(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.z_layer = Linear(ae_n_enc_3, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z))
        z_ae = self.z_layer(z)
        return z_ae


class AE_decoder(nn.Module):

    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))
        x_hat = self.x_bar_layer(z)
        return x_hat


class AE(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

    def forward(self, x):
        z_ae = self.encoder(x)
        x_hat = self.decoder(z_ae)
        return x_hat, z_ae


class attn_head(Module):
    def __init__(self,w1,w2,in_drop=0.0, coef_drop=0.0, activation=nn.Tanh):
        super(attn_head, self).__init__()

        self.act = activation
        self.in_dropout = Dropout(in_drop)
        self.coef_dropout = Dropout(coef_drop)
        self.w1 = w1
        self.w2 = w2


    def forward(self, seq, bias_mat, active = False):

        seq = self.in_dropout(seq)  # h  d = 8
        seq_fts = self.act(F.linear(seq,self.w1))  # K   d = 8
        seq_fts2 = self.act(F.linear(seq,self.w2) )# Q   d = 8  公式(4)

        logits = torch.matmul(seq_fts2, seq_fts.t()) / np.sqrt(
                seq_fts.shape[-1])  # Q^T * K / sqrt(d)

        coefs =torch.exp(logits) * bias_mat  # adj or kernel T    公式(5)
        sum_coefs = torch.sum(coefs, dim=-1)
        coefs = coefs / sum_coefs.sum().item()
        seq_fts = self.in_dropout(seq_fts)  # V   公式（7）

        if active:
            seq_fts = self.act(seq_fts)
        ret= torch.matmul(coefs, seq_fts)
        ret = torch.matmul(bias_mat,ret)   #公式（8）

        return ret  # , coefs, p_mat

class GNNLayer(Module):

    def __init__(self, in_features, out_features,n_head):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()
        self.s = nn.Sigmoid()
        self.atts = []
        self.n_head = n_head
        self.w11 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w21 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w12 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w22 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w13 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w23 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w14 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w24 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w15 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w25 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w16 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w26 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w17 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w27 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w18 = Parameter(torch.FloatTensor(out_features, in_features))
        self.w28 = Parameter(torch.FloatTensor(out_features, in_features))



        torch.nn.init.xavier_uniform_(self.w11)
        torch.nn.init.xavier_uniform_(self.w21)
        torch.nn.init.xavier_uniform_(self.w12)
        torch.nn.init.xavier_uniform_(self.w22)
        torch.nn.init.xavier_uniform_(self.w13)
        torch.nn.init.xavier_uniform_(self.w23)
        torch.nn.init.xavier_uniform_(self.w14)
        torch.nn.init.xavier_uniform_(self.w24)
        torch.nn.init.xavier_uniform_(self.w15)
        torch.nn.init.xavier_uniform_(self.w25)
        torch.nn.init.xavier_uniform_(self.w16)
        torch.nn.init.xavier_uniform_(self.w26)
        torch.nn.init.xavier_uniform_(self.w17)
        torch.nn.init.xavier_uniform_(self.w27)
        torch.nn.init.xavier_uniform_(self.w18)
        torch.nn.init.xavier_uniform_(self.w28)


        if n_head > 0:
            in_drop = 0.1
            coef_drop = 0.1
        else:
            in_drop = 0
            coef_drop = 0

        if n_head == 1:
          self.atts.append(attn_head(self.w11,self.w21,in_drop, coef_drop,self.act))
        elif n_head == 2:
            self.atts.append(attn_head(self.w11, self.w21, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w12, self.w22, in_drop, coef_drop, self.act))
        elif n_head == 4:
            self.atts.append(attn_head(self.w11, self.w21, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w12, self.w22, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w13, self.w23, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w14, self.w24, in_drop, coef_drop, self.act))
        elif n_head == 8:
            self.atts.append(attn_head(self.w11, self.w21, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w12, self.w22, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w13, self.w23, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w14, self.w24, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w15, self.w25, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w16, self.w26, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w17, self.w27, in_drop, coef_drop, self.act))
            self.atts.append(attn_head(self.w18, self.w28, in_drop, coef_drop, self.act))


    def forward(self, features, adj, active):


        output = torch.zeros(features.shape[0],self.out_features).to("cuda")
        for head in self.atts:

            output += head(features, adj,active)/self.n_head

        return output


class IGAE_encoder(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,n_head, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1,n_head[0])
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2,n_head[1])
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3, n_head[2])
        self.s = nn.Sigmoid()

    def forward(self, x, adj):

        z = self.gnn_1(x, adj, active=True)
        z = self.gnn_2(z, adj, active=True)
        z_igae = self.gnn_3(z, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class IGAE_decoder(nn.Module):

    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,n_head, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2,n_head[0])
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3,n_head[1])
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input,n_head[2])
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z = self.gnn_4(z_igae, adj, active=True)
        z = self.gnn_5(z, adj, active=True)
        z_hat = self.gnn_6(z, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class IGAE(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_head = [4,1,1],
            n_input=n_input)
# 注意力头默认  n_head = [4,1,1]
        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_head= [4,1,1],
            n_input=n_input)

# 注意力头默认  n_head = [4,1,1]
    def forward(self, x, adj):

        z_igae, z_igae_adj = self.encoder(x, adj)
        z_hat, z_hat_adj = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat


class q_distribution(nn.Module):
    def __init__(self, centers):
        super(q_distribution, self).__init__()
        self.cluster_centers = centers
        self.v = opt.args.freedom_degree

    def forward(self, z, z_ae, z_igae):
        q = 1.0 / (1.0 + torch.sum(torch.pow((z).unsqueeze(1) - self.cluster_centers, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_centers, 2), 2) / self.v)
        q_ae = q_ae.pow((self.v + 1.0) / 2.0)
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_igae = 1.0 / (1.0 + torch.sum(torch.pow(z_igae.unsqueeze(1) - self.cluster_centers, 2), 2) / self.v)
        q_igae = q_igae.pow((self.v + 1.0) / 2.0)
        q_igae = (q_igae.t() / torch.sum(q_igae, 1)).t()
        return [q, q_ae, q_igae]
