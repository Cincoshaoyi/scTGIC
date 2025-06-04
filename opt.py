import argparse

parser = argparse.ArgumentParser(description='scTGIC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
parser.add_argument('--name', type=str, default="inhouse")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--rec_epoch', type=int, default=30)
parser.add_argument('--fus_epoch', type=int, default=200)
parser.add_argument('--epoch', type=int, default=600)
parser.add_argument('--pretrain', type=bool, default=False)

# parameters
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--alpha_value', type=float, default=0.1)
parser.add_argument('--lambda1', type=float, default=10)
parser.add_argument('--lambda2', type=float, default=0.01)
parser.add_argument('--lambda3', type=float, default=10)
parser.add_argument('--method', type=str, default='euc')
parser.add_argument('--first_view', type=str, default='ATAC')
parser.add_argument('--lr', type=float, default=5e-3)
#后补进来   V
parser.add_argument('--freedom_degree', type=float, default=1.0)

# dimension of input and latent representations
parser.add_argument('--n_d1', type=int, default=50)
parser.add_argument('--n_d2', type=int, default=10)
parser.add_argument('--n_z', type=int, default=20)

# AE structure parameter
parser.add_argument('--ae_n_enc_1', type=int, default=512)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=128)
parser.add_argument('--ae_n_dec_1', type=int, default=128)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=512)

# IGAE structure parameter
parser.add_argument('--gae_n_enc_1', type=int, default=256)
parser.add_argument('--gae_n_enc_2', type=int, default=128)
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=128)
parser.add_argument('--gae_n_dec_3', type=int, default=256)

# clustering performance: acc, nmi, ari, f1
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--fmi', type=float, default=0)
parser.add_argument('--hom', type=float, default=0)
parser.add_argument('--com', type=float, default=0)
parser.add_argument('--ami', type=float, default=0)

args = parser.parse_args()
