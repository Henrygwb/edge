# Implementation of additive GP with multi-level kernel functions.
# Additive Gaussian process with two kernels - data (trajectory/individual) level and feature level.
# SoR approximation of K_{xx}.
# Parameters that solved by gradient descent: NN/embedding parameters, kernel parameters, Z.
# Parameters that solved by letting gradient equals to zero: \mu, L.

import torch
import argparse
import numpy as np
from scipy import io
import seaborn as sns
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, trainX, trainy, trainId, trainOid):
        self.trainX = trainX
        self.trainy = trainy
        self.trainId = trainId
        self.trainOid = trainOid

    def __getitem__(self, i):
        return self.trainX[i], self.trainy[i], self.trainId[i], self.trainOid[
            i]  # the last index is the observation index

    def __len__(self):
        return len(self.trainy)


# Define the RBF kernel operation.
class RBFKernel(nn.Module):
    def __init__(self, input_dim):
        super(RBFKernel, self).__init__()
        self.input_dim = input_dim
        self.log_std = nn.Parameter(torch.zeros([1]))
        self.log_ls = nn.Parameter(torch.zeros([self.input_dim]))

    def _square_scaled_dist(self, X, Z=None):
        ls = self.log_ls.exp() # exp(self.log_ls)
        scaled_X = X / ls[None, :]
        scaled_Z = Z / ls[None, :]
        X2 = scaled_X.pow(2).sum(1, keepdim=True) # X^2
        Z2 = scaled_Z.pow(2).sum(1, keepdim=True)
        XZ = scaled_X @ scaled_Z.t()
        r2 = X2 - 2 * XZ + Z2.t()
        return r2.clamp(min=0) # clip

    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        assert X.shape[1] == Z.shape[1]
        base = -0.5 * self._square_scaled_dist(X, Z)
        base += 2 * self.log_std
        return base.clamp(min=-10, max=10).exp()


def kernelMix(k1, k2):
    return k1 + k2


# Encoders for the kernel functions.
class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.CELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.CELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, x):
        return self.net(x)


# Kernel layer with two kernel functions, the forward function gives the final covariance (addition of two kernels).
class DeepRBFKernel(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim, num_indv, n_induce, indv_dim):
        super(DeepRBFKernel, self).__init__()
        self.encoder = Encoder(input_dim, z_dim, hidden_dim)
        self.rbf_indv = RBFKernel(indv_dim) # data dim
        self.rbf_obsr = RBFKernel(z_dim)
        self.indv_dim = indv_dim
        self.num_indv = num_indv
        self.indv_embedding = nn.Parameter(torch.rand([num_indv + n_induce, indv_dim])) # learnable data embedding.

    def forward(self, X, xid, Z=None, zid=None):
        embed_X = self.encoder(X)
        embed_Xid = self.indv_embedding[xid]
        if Z is None:
            k1 = self.rbf_indv(embed_Xid, embed_Xid)
            k2 = self.rbf_obsr(embed_X, embed_X)
            return kernelMix(k1, k2)
        else:
            embed_Zid = self.indv_embedding[zid + self.num_indv]
            k1 = self.rbf_indv(embed_Xid, embed_Zid)
            k2 = self.rbf_obsr(embed_X, Z)
            return kernelMix(k1, k2)

    def indv_kernel(self, xid):
        embed_Xid = self.indv_embedding[xid]
        return self.rbf_indv(embed_Xid, embed_Xid)

    def rbf_direct(self, Z, zid):
        embed_Zid = self.indv_embedding[zid + self.num_indv]
        k1 = self.rbf_indv(embed_Zid, embed_Zid)
        k2 = self.rbf_obsr(Z, Z)
        return kernelMix(k1, k2)


# The full model.
class SparseGPRegression(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim, n_induce, num_indv, indv_dim):
        super(SparseGPRegression, self).__init__()
        self.n_induce = n_induce
        self.input_dim = input_dim
        self.deepkernel = DeepRBFKernel(input_dim, z_dim, hidden_dim, num_indv, n_induce,indv_dim=indv_dim)
        self.embed_Z = nn.Parameter(torch.rand([n_induce, z_dim])) # inducing points at the latent layer.
        self.log_sigma = nn.Parameter(torch.zeros(1)) # Independent observation error: N(0, \sigma^2).

    def forward(self, x, y, xid, u, L):
        # Compute the ELBO. Variational distribution u ~ N(u, LL^T)
        sigma2 = self.log_sigma.exp().pow(2)
        zid = torch.arange(self.n_induce).type(torch.long).to(x.device)
        kxz, kzz, kxx, Lz, kzz_inv, first = kernel_compute(x, xid, self.embed_Z, zid, self.deepkernel)
        m = first @ u
        v = first @ L
        Sigma_q = L @ L.T

        # Log likelihood term.
        # KL divergence term.
        l1 = 1 / sigma2 * (y.T @ y - 2 * y.T @ m + m.T @ m + (v.T @ v).sum()) # loglikihood term
        l2 = (Lz.diag().log().sum() - L.diag().log().sum()) * 2 + \
             (kzz_inv @ Sigma_q).trace() + (u.T @ kzz_inv @ u)
        return l1 / 2, l2 / 2

    def predict_logit(self, x, xid, u):
        # Compute the predictive mean given u.
        zid = torch.arange(self.n_induce).type(torch.long).to(x.device)
        kxz, kzz, kxx, Lz, kzz_inv, first = kernel_compute(x, xid, self.embed_Z, zid, self.deepkernel)
        m, _ = conditional_dist(kxz, kxx, first, u.view(-1, 1))
        return m.view(-1)


def kernel_compute(X, xid, Z, zid, kernel):
    kxz = kernel(X, xid, Z, zid)
    kzz = kernel.rbf_direct(Z, zid)
    kxx = kernel(X, xid)
    kzz = kzz + torch.eye(len(kzz)).to(kzz.device) * factor
    Lz = torch.cholesky(kzz)
    inv_Lz = torch.inverse(Lz)
    kzz_inv = inv_Lz.t() @ inv_Lz # K_{zz}^{-1}
    first = kxz @ kzz_inv # K_{xz}K_{zz}^{-1}
    return kxz, kzz, kxx, Lz, kzz_inv, first


# posterior conditional distribution of f|u ~ N(K_{xz}K_{zz}^{-1}u, K_{xx} - K_{xz}K_{zz}^{-1}K_{xz}^{t})
def conditional_dist(kxz, kxx, first, u):
    mean = first @ u  # nxc
    var = kxx - first @ kxz.t()  # nxn
    return mean, var


# Compute the optimal L and u given the current value of other parameters.
# This function can be replaced with stochastic gradient descent if the optimal solution of L and u are not solvable.
def solve_posterior(sgp):
    zid = torch.arange(sgp.n_induce).type(torch.long)
    beta = 1 / sgp.log_sigma.exp().pow(2) # \sigma^{-2}
    x = trainX
    y = trainy
    xid = trainId
    if args.cuda:
        x = trainX.cuda()
        y = trainy.cuda()
        xid = trainId.cuda()
        zid = zid.to('cuda')
    kxz, kzz, kxx, Lz, kzz_inv, first = kernel_compute(x, xid, sgp.embed_Z, zid, sgp.deepkernel)
    B = kzz + beta * kxz.T @ kxz
    B_chol = torch.cholesky(B)
    B_chol_inv = torch.inverse(B_chol)
    B_inv = B_chol_inv.T @ B_chol_inv
    A = kzz @ B_inv
    u = beta * A @ kxz.T @ y
    Sigma = A @ kzz
    U = torch.cholesky(Sigma + torch.eye(len(Sigma)).to(Sigma.device) * factor) # adding a small number on the diag for more stable computing.
    P = (torch.ones([sgp.n_induce, sgp.n_induce]) + torch.eye(sgp.n_induce)).to(x.device)
    P = torch.cholesky(P)
    L = (U.diag() / P.diag()).diag()
    return u, L



def train(args):
    valid_r2_past = -1e10
    count = 0
    for epoch in range(args.epochs):
        if epoch == 0:
            sgp.eval()
            with torch.no_grad():
                u, L = solve_posterior(sgp)
        sgp.train()
        epoch_loss = 0.
        preds = []
        ys = []
        for i, (x, y, xid, oid) in enumerate(train_loader):
            ys.extend(y.numpy())
            x = x.type(torch.float)
            y = y.type(torch.float)
            xid = xid.type(torch.long)
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
                xid = xid.cuda()
            optimizer.zero_grad()
            l1, l2 = sgp(x, y, xid, u, L)
            loss = l1 + l2
            pred = sgp.predict_logit(x, xid, u)
            preds.extend(pred.detach().cpu().numpy())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
        ys, preds = np.array(ys), np.array(preds)
        print(f'epoch {epoch}: r2 -> {r2_score(ys, preds)}, rmse -> {np.sqrt(np.mean((ys - preds) ** 2))}')

        u,L,valid_r2 = test(args,True,u,L)
        if valid_r2 < valid_r2_past:
            count += 1
        else:
            count = 0
        valid_r2_past = valid_r2

        if epoch % args.test_frequency == 0:
            if epoch > 0:
                u,L,final_test = test(args,False,None,None)
            else:
                u,L,final_test = test(args,False, u, L)

        if count >= args.valid_dec_count:
            break
    return final_test


def test(args,isValid,u=None,L=None):
    sgp.eval()
    if u is None:
        with torch.no_grad():
            u, L = solve_posterior(sgp)

    test_loss = 0.
    preds = []
    ys = []
    if isValid:
        loader = valid_loader
    else:
        loader = test_loader
    for i, (x, y, xid, oid) in enumerate(loader):
        ys.extend(y.numpy())
        x = x.type(torch.float)
        y = y.type(torch.float)
        xid = xid.type(torch.long)
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
            xid = xid.cuda()
        with torch.no_grad():
            l1, l2 = sgp(x, y, xid, u, L)
            loss = l1 + l2
            test_loss += loss.item()

            pred = sgp.predict_logit(x, xid, u)
            preds.extend(pred.cpu().numpy())
    # report test diagnostics
    normalizer_test = len(loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    print(f"\t average {'valid' if isValid else 'test'} loss: %.4f" % (total_epoch_loss_test))
    r2 = r2_score(ys, preds)
    print(f"\t {'valid' if isValid else 'test'} r2: {r2}")
    return u,L,r2


def arge_parser():
    fn = 'tadpole_small'
    parser = argparse.ArgumentParser(description='L-DKGPR')
    # parser.add_argument('--file', type=str, help='location of the input file, should be a .mat file',default='..//cluster3_1.mat')
    parser.add_argument('--file', type=str, help='location of the input file, should be a .mat file',default=f'data/{fn}.mat')
    parser.add_argument('--batch_size', type=int, help='minibatch size for training, default 1024',
                        default=1024)
    parser.add_argument('--dropout', type=float, help='dropout rate, default 0.2', default=0.2)
    parser.add_argument('--D', type=int, help='latent dimensions, default 10',default=10)
    parser.add_argument('--hidden_dim',type=int, help='number of hidden units, default 16', default=16)
    parser.add_argument('--factor',type=float,help='the value added to diagonal of correlation matrix to avoid singularity, default 0.0001; If singular problem exists, please choose a larger value.', default=1e-2)
    parser.add_argument('--M', type=int, help='the number of inducing points, default 10', default=10)
    parser.add_argument('--lr', type=float,
                        help='learning rate for Theta - phi, default 0.001. Decrease lr if bad performance is achieved', default=1e-3)
    parser.add_argument('--lr_indv_embedding', type=float,
                        help='learning rate for phi, default 0.01. Decrease lr if bad performance is achieved', default=1e-1)
    parser.add_argument('--epochs', type=int,
                        help='learning epochs for training, default 300', default=300)
    parser.add_argument('--cuda', dest='cuda',
                        help='use gpu', action='store_true')
    parser.add_argument('--cpu', dest='cuda', help='use cpu only', action='store_false')
    parser.add_argument('--test_frequency', type=int,
                        help='output the test loss in every {test_frequency} epochs, default 5', default=5)
    parser.add_argument('--save_path', type=str,
                        help="the path to the saved model after training.", default=f'saveModels/{fn}')
    parser.add_argument('--load_path', type=str,
                        help="load a saved model If specified, then no training is performed. default 'None'",
                        # default='saveModels/lastRun')
                        default = 'None')
    parser.add_argument('--seed', type=int, help='random seed, default 0', default=0)
    parser.add_argument('--valid_dec_count', type=int, help='tolerance for time of decreasing r2 on validation set', default=10)
    parser.add_argument('--number_cluster',type=int,help='number of cluster in correlation plot. Only valid in real-life data. default 2.',default=3)
    args = parser.parse_args()
    return args


def visualizeCorr(sgp,args):
    sgp.cpu()
    if args.file.split('/')[-2] == 'simulation':
        final_corr = data['corr']
        allX = torch.tensor(data['data']).type(torch.float)
        allIid = data['iid'].reshape(-1)

        plt.figure()
        sns.heatmap(
            final_corr,
            cmap="YlGnBu",
            square=True,
            robust=True,
            xticklabels=False,
            yticklabels=False,
        )

        corr = sgp.deepkernel(allX, allIid).detach().cpu().numpy()
        plt.figure()
        sns.heatmap(
            corr,
            cmap='YlGnBu',
            square=True,
            robust=True,
            xticklabels=False,
            yticklabels=False,
        )
        plt.show()
    else:
        from sklearn.cluster import SpectralCoclustering
        indv_corr = sgp.deepkernel.indv_kernel(torch.arange(len(idMap))).detach().cpu().numpy()
        num_c = args.number_cluster
        model = SpectralCoclustering(n_clusters=num_c, random_state=0)
        model.fit(indv_corr)
        fit_data = indv_corr[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.row_labels_)]
        rows = np.random.permutation(np.arange(len(fit_data)))
        rows = rows[:3300]
        rows = np.sort(rows)
        clusterRes = model.row_labels_
        cl = np.argsort(clusterRes)
        ax = sns.heatmap(
            indv_corr[cl][:, cl],
            cmap='YlGnBu',
            square=True,
            robust=True,
            xticklabels=False,
            yticklabels=False,
        )
        plt.show()

if __name__ == '__main__':
    args = arge_parser()
    file = args.file
    factor = args.factor
    dropout_rate = args.dropout
    # the mat file should contains the following fields:
    #       trainId, trainOid, trainX, trainY; testId, testOid, testX, testY;
    data = io.loadmat(file)
    trainX = torch.from_numpy(data['trainX']).type(torch.float)
    trainy = torch.from_numpy(data['trainY'].reshape(-1)).type(torch.float)
    testX = torch.from_numpy(data['testX']).type(torch.float)
    testy = torch.from_numpy(data['testY'].reshape(-1)).type(torch.float)

    trainId_Ori = data['trainId'].reshape(-1)
    testId_Ori = data['testId'].reshape(-1)
    trainOid_Ori = data['trainOid'].reshape(-1)
    testOid_Ori = data['testOid'].reshape(-1)
    allX = torch.cat([trainX, testX], dim=0)
    ally = torch.cat([trainy, testy], dim=0)
    ids = set(list(np.concatenate([trainId_Ori, testId_Ori])))
    oids = set(list(np.concatenate([trainOid_Ori, testOid_Ori])))
    idMap = {}
    i = 0
    for x in ids:
        idMap[x] = i
        i += 1
    trainId = torch.FloatTensor([idMap[x] for x in trainId_Ori]).type(torch.long)
    testId = torch.FloatTensor([idMap[x] for x in testId_Ori]).type(torch.long)
    minOid = np.min(list(oids))
    trainOid = torch.FloatTensor(trainOid_Ori - minOid)
    testOid = torch.FloatTensor(testOid_Ori - minOid)
    allIid = torch.cat([trainId, testId])
    allmean = ally.mean()
    trainy -= allmean
    testy -= allmean

    # split train into train and valid
    tr_idx = np.arange(len(trainX))
    train_idx, valid_idx = train_test_split(tr_idx,test_size=int(2/7 * len(trainX)), random_state=args.seed)
    train_ds = MyDataset(trainX[train_idx],trainy[train_idx],trainId[train_idx],trainOid[train_idx])
    valid_ds = MyDataset(trainX[valid_idx],trainy[valid_idx],trainId[valid_idx],trainOid[valid_idx])
    test_ds = MyDataset(testX, testy, testId, testOid)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    sgp = SparseGPRegression(trainX.shape[1], args.D, args.hidden_dim, args.M, len(ids),args.D)
    params = []
    for n, p in sgp.named_parameters():
        if n.startswith('embed_Z') or n.startswith('deepkernel.indv_embedding'):
            params.append({'name': n, 'params': p, 'lr': args.lr_indv_embedding})
        else:
            params.append({'name': n, 'params': p, 'lr': args.lr})

    if args.cuda:
        sgp = sgp.to('cuda')
    optimizer = Adam(params, lr=args.lr)
    if args.load_path is 'None':
        r2 = train(args)
        if args.save_path is not 'None':
            torch.save(sgp.state_dict(), args.save_path)
        print(f'the final test r2 {r2}')
    else:
        sgp.load_state_dict(torch.load(args.load_path))

    visualizeCorr(sgp, args)
