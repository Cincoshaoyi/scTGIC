import opt
from encoder import *


class scTGIC(nn.Module):
    def __init__(self, ae1, ae2, gae1, gae2, n_node=None, device=None):
        super(scTGIC, self).__init__()

        self.device = device

        self.ae1 = ae1
        self.ae2 = ae2

        self.gae1 = gae1
        self.gae2 = gae2

        self.a = nn.Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z).to(self.device),0.5),
                           requires_grad=True) # Z_ae, Z_igae
        self.b = nn.Parameter((1 - self.a).to(self.device), requires_grad=True)

      #  self.alpha = Parameter(torch.zeros(1))  # ZG, ZL

        self.cluster_centers1 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z).to(self.device), requires_grad=True)
        self.cluster_centers2 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z).to(self.device), requires_grad=True)
        self.projection = nn.Linear(2 * opt.args.n_z, opt.args.n_z)  # 添加投影层：2*n_z -> n_z
        torch.nn.init.xavier_normal_(self.cluster_centers1.data)
        torch.nn.init.xavier_normal_(self.cluster_centers2.data)
        self.q_distribution1 = q_distribution(self.cluster_centers1)
        self.q_distribution2 = q_distribution(self.cluster_centers2)

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(n_node, opt.args.n_clusters).to(self.device),
            nn.Softmax(dim=1)
        )


        self.gamma = Parameter(torch.zeros(1).to(self.device))

    def emb_fusion(self, adj, z_ae, z_igae):
        z_ae = z_ae.to(self.device)
        z_igae = z_igae.to(self.device)
        a = self.a.to(self.device)
        b = self.b.to(self.device)
        z_i = self.a * z_ae + self.b * z_igae
        z_tilde = torch.mm(adj.to(self.device), z_i)


        return z_tilde

    def forward(self, x1, adj1, x2, adj2, pretrain=False):
        # node embedding encoded by AE
        z_ae1 = self.ae1.encoder(x1)
        z_ae2 = self.ae2.encoder(x2)

        # node embedding encoded by IGAE
        z_igae1, a_igae1 = self.gae1.encoder(x1, adj1)
        z_igae2, a_igae2 = self.gae2.encoder(x2, adj2)

        z1 = self.emb_fusion(adj1, z_ae1, z_igae1)
        z2 = self.emb_fusion(adj2, z_ae2, z_igae2)

        z1_tilde = self.label_contrastive_module(z1.T)
        z2_tilde = self.label_contrastive_module(z2.T)

        cons = [z1, z2, z1_tilde, z2_tilde]

        # AE decoding
        x_hat1 = self.ae1.decoder(z1)
        x_hat2 = self.ae2.decoder(z2)

        # IGAE decoding
        z_hat1, z_adj_hat1 = self.gae1.decoder(z1, adj1)
        a_hat1 = a_igae1 + z_adj_hat1

        z_hat2, z_adj_hat2 = self.gae2.decoder(z2, adj2)
        a_hat2 = a_igae2 + z_adj_hat2

        if not pretrain:
            # the soft assignment distribution Q
            Q1 = self.q_distribution1(z1, z_ae1, z_igae1)
            Q2 = self.q_distribution2(z2, z_ae2, z_igae2)
        else:
            Q1, Q2 = None, None

        return x_hat1, z_hat1, a_hat1, x_hat2, z_hat2, a_hat2, Q1, Q2, z1, z2, cons
