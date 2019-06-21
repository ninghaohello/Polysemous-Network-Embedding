import os, pickle, random, argparse, torch
import numpy as np
import scipy as sp
from scipy import io as sio
from sklearn.decomposition import NMF
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from poly_pte import Word2Vec, PolyPTE
from myutils import normalizeWH


def bipartite2homo(fn_ratingList, index_shift):
    ratingList = sio.loadmat(fn_ratingList)
    for n in range(len(ratingList['items'][0])):
        ratingList['items'][0][n] += index_shift
    return ratingList

def shift_to_zero(ratingList):
    if np.min(ratingList['users'][0]) == 1:
        ratingList['users'][0] -= 1
        ratingList['items'][0] -= 1
    return ratingList

def clustering(fn_sparseMat, name_data, K=5):
    mat_ratings = sp.sparse.load_npz(fn_sparseMat)
    model = NMF(n_components=K, init='random', random_state=0, alpha=0.05)
    W = model.fit_transform(mat_ratings)
    H = model.components_
    np.save(os.path.join(os.path.dirname(os.getcwd()), 'data', name_data, 'W_'+str(K)+'.npy'), W)
    np.save(os.path.join(os.path.dirname(os.getcwd()), 'data', name_data, 'H_'+str(K)+'.npy'), H)
    return W, H

def prob_cluster(WH):
    pnk = np.copy(WH)
    for n in range(WH.shape[0]):
        if np.sum(pnk[n]) == 0:
            pnk[n] = np.ones(pnk[n].shape)
        pnk[n] /= np.sum(pnk[n])
    return pnk


class PermutedSubsampledCorpus(Dataset):
    def __init__(self, data, ws=None):
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords= self.data[idx]
        return iword, np.array(owords)


def run_polypte(args, dataset, vocab1_size, vocab2_size):
    # Initiate or reload model
    name_this_run = args.name_model
    name_append = str(args.e_dim) + '_' + str(args.K) + '_' + str(args.num_negs) + '_' + str(args.num_samples_clustering)
    model = Word2Vec(vocab1_size=vocab1_size, vocab2_size=vocab2_size,
                     num_poly=args.K, embedding_size=args.e_dim)        # embeddings
    modelpath = os.path.join(args.save_dir, '{}.pt'.format(name_this_run))
    polypte = PolyPTE(embedding=model, vocab1_size=vocab1_size, vocab2_size=vocab2_size,
                      num_poly=args.K, num_negs=args.num_negs)          # learning module
    if os.path.isfile(modelpath) and args.conti:
        print('Old model loaded...')
        polypte.load_state_dict(torch.load(modelpath))
    if args.cuda:
        print('Using cuda...')
        polypte = polypte.cuda()
    optim = Adam(polypte.parameters())
    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(name_this_run))
    if os.path.isfile(optimpath) and args.conti:
        print('Old Optim loaded...')
        optim.load_state_dict(torch.load(optimpath))

    # Training
    for epoch in range(1, args.epoch + 1):
        dataset = PermutedSubsampledCorpus(dataset)
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)

        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for iword, owords in pbar:
            loss = polypte(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())

        # Get final embeddings and save
        embedding_1 = model.vectors_1.weight.data.cpu().numpy()
        embedding_1 = np.reshape(embedding_1, (vocab1_size, args.K, args.e_dim))
        pickle.dump(embedding_1, open(os.path.join(args.data_dir, 'polypte_embedding1_'+name_append+'.dat'), 'wb'))
        embedding_2 = model.vectors_2.weight.data.cpu().numpy()
        embedding_2 = np.reshape(embedding_2, (vocab2_size, args.K, args.e_dim))
        pickle.dump(embedding_2, open(os.path.join(args.data_dir, 'polypte_embedding2_'+name_append+'.dat'), 'wb'))

    torch.save(polypte.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(name_this_run)))
    torch.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(name_this_run)))


if __name__ == "__main__":
    name_data = 'movielens'
    poly = 5

    parser = argparse.ArgumentParser()
    parser.add_argument('--name_model', type=str, default='polypte', help="model name")
    parser.add_argument('--K', type=int, default=poly, help="number of clusters")
    parser.add_argument('--e_dim', type=int, default=int(150 / poly), help="embedding dimension")
    parser.add_argument('--num_negs', type=int, default=30, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=2, help="number of epochs")
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--conti', action='store_true', help="continue learning")  # 'store_true' means 'false'
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--clustering', action='store_false', help="do clustering first or not")
    parser.add_argument('--num_samples_clustering', type=int, default=poly * poly,
                        help="how many node cluster samples for each link")
    parser.add_argument('--path_length', type=int, default=3, help="length of path (3 for recSys)")
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.getcwd()), 'data', name_data),
                        help="data directory path")
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.path.dirname(os.getcwd()), 'data', name_data),
                        help="model directory path")

    args_training = parser.parse_args()

    fn_sparseMat = os.path.join(os.path.dirname(os.getcwd()), 'data', name_data, 'sparsemat.npz')
    fn_training = os.path.join(os.path.dirname(os.getcwd()), 'data', name_data, 'training.mat')
    fn_testing = os.path.join(os.path.dirname(os.getcwd()), 'data', name_data, 'testing.mat')

    # Some adjustment
    mat_ratings = sp.sparse.load_npz(fn_sparseMat)
    num_users, num_items = mat_ratings.shape
    print("Number of users: %d; Number of items: %d" % (num_users, num_items))
    del mat_ratings
    training_data = bipartite2homo(fn_training, num_users)  # concatenate index
    testing_data = bipartite2homo(fn_testing, num_users)
    training_data = shift_to_zero(training_data)
    testing_data = shift_to_zero(testing_data)

    # Clustering
    if args_training.clustering:
        W, H = clustering(fn_sparseMat, name_data, K=args_training.K)
    else:
        W = np.load(os.path.join(os.path.dirname(os.getcwd()), 'data', name_data, 'W_' + str(args_training.K) + '.npy'))
        H = np.load(os.path.join(os.path.dirname(os.getcwd()), 'data', name_data, 'H_' + str(args_training.K) + '.npy'))
    prob_W, prob_H = normalizeWH(W, H)      # be careful the mearning of this function, not probability, but likelihood
    prob_W = prob_cluster(prob_W)
    prob_H = prob_cluster(prob_H.T).T

    # Sample all node pairs (training samples) from graph
    uir = training_data
    num_ratings = uir['ratings'].shape[1]
    dataset = []
    if args_training.epoch > 0:
        for i in range(num_ratings):
            user, item = uir['users'][0, i], uir['items'][0, i]
            item = item - num_users     # de-concatenate index
            p_joint = (prob_W[user] + prob_H[:,item])/2
            for kk in range(args_training.num_samples_clustering):
                k1 = np.random.choice(args_training.K, 1, p=p_joint)[0]  # which cluster prior
                k2 = np.random.choice(args_training.K, 1, p=p_joint)[0]  # which cluster prior
                dataset.append((k1+user*args_training.K, [k2+item*args_training.K]))
        print("Sampling done")

    # Run embedding algorithm
    if args_training.epoch > 0:
        run_polypte(args_training, dataset, num_users, num_items)