import numpy as np
import scipy as sp
import os, pickle, random, datetime
from scipy import io as sio
from scipy.sparse import csr_matrix
from six import iterkeys
from collections import defaultdict
from gensim.models import Word2Vec


class Graph(defaultdict):
    def __init__(self):
        super().__init__(list)

    def nodes(self):
        return self.keys()

    def make_consistent(self):
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))
        self.remove_self_loops()
        return self

    def remove_self_loops(self):
        for x in self:
            if x in self[x]:
                self[x].remove(x)
        return self

    def random_walk_poly(self, num_poly, pnk, path_length, alpha = 0, rand = random.Random(), start = None):
        G = self

        if start:
            path = [start]
            p_context = np.copy(pnk[start])
        else:
            start = rand.choice(list(G.keys()))
            path = [start]
            p_context = np.copy(pnk[start])

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    node_new = rand.choice(G[cur])
                    path.append(node_new)
                    p_context += pnk[node_new]
                else:
                    path.append(path[0])
                    p_context += pnk[path[0]]
            else:
                break

        p_context /= len(path)

        ps_min = []
        for n in path:
            tmp = np.array(np.minimum(pnk[n], p_context))
            ps_min.append(tmp/np.sum(tmp))

        walks = []
        for i in range(num_sample_per_walk):
            ks = [np.random.choice(num_poly, 1, p=ps_min[i])[0] for i in range(len(path))]
            walks.append([str(node) + '_' + str(k) for node, k in zip(path, ks)])

        return walks, p_context


class Poly_DeepWalk():
    def __init__(self, num_nodes, dim, num_poly, fn_result, fn_clustering, fn_testing,
                 appendix_this_run, appendix_path_this_run, window_size,
                 path_length, num_neg):
        self.num_nodes = num_nodes
        self.dim = dim
        self.num_poly = num_poly
        self.path_length = path_length
        self.window_size = window_size
        self.num_neg = num_neg
        self.fn_result = fn_result
        self.fn_clustering = fn_clustering
        self.fn_testing = fn_testing
        self.appendix_this_run = appendix_this_run
        self.appendix_path_this_run = appendix_path_this_run

        self.num_embds = self.num_nodes * self.num_poly

        self.cluster = sio.loadmat(self.fn_clustering)['H']
        self.pnk = self.prob_cluster()

        self.flag_empirical = True
        self.pnk_emp = np.zeros(self.pnk.shape)
        self.pnk_emp_counter = np.zeros(self.num_nodes)


    def prob_cluster(self):
        pnk = np.copy(self.cluster)
        for n in range(self.num_nodes):
            if np.sum(pnk[n]) == 0:
                pnk[n] = np.ones(pnk[n].shape)
            pnk[n] /= np.sum(pnk[n])
        return pnk


    def generate_walks_batch(self, G, num_walks, alpha = 0,
                             rand=random.Random()):
        walks = []
        nodes = list(G.nodes())
        print("Total number of nodes: %d" % len(nodes))
        print("The estimated number of embeddings: %d" % (len(nodes)*self.num_poly))

        for cnt in range(num_walks):
            if cnt%10 == 0:
                print(cnt)
            rand.shuffle(nodes)
            for node in nodes:
                walk, p_context = G.random_walk_poly(self.num_poly, self.pnk, self.path_length, alpha=alpha, start=node)
                walks.extend(walk)

                if self.flag_empirical:
                    for w in walk:
                        for nk in w:
                            n, k = nk.split('_')
                            n = int(n)
                            self.pnk_emp[n] += p_context
                            self.pnk_emp_counter[n] += 1

        return walks


    def train(self, G, num_walks_per_node):
        print("Generating walks...")
        walks = self.generate_walks_batch(G, num_walks=num_walks_per_node,
                                          alpha=0)

        print("Training...")
        model = Word2Vec(sentences=walks,
                         size=self.dim,
                         window=self.window_size,
                         min_count=0,
                         sg=1, hs=1,
                         negative=self.num_neg,
                         workers=1)
        print("Saving embedding...")
        model.wv.save_word2vec_format(self.fn_result)


def load_edgelist(ratingList, undirected = True):
    G = Graph()
    num_ratings = len(ratingList['vals'][0])
    for n in range(num_ratings):
        u = ratingList['rows'][0][n]
        i = ratingList['cols'][0][n]
        G[u].append(i)
        if undirected:
            G[i].append(u)

    G.make_consistent()
    return G


if __name__ == '__main__':
    name_dataset = 'BlogCatalog'
    num_sample_per_walk = 10
    dim = 150
    num_poly = 6
    window_size = 4
    path_length = 11
    num_neg = 15
    num_walks_per_node = 110

    folder_dataset = os.path.join(os.path.dirname(os.getcwd()), 'data', name_dataset)
    appendix_this_run = str(dim) + '_' + str(num_poly) + '_' + str(window_size) + '_' + str(num_neg) + '_' + str(num_walks_per_node)# + '_' + str(num_sample_per_walk)
    appendix_path_this_run = str(window_size) + '_' + str(num_walks_per_node)
    fn_training = os.path.join(folder_dataset, 'training.mat')
    fn_testing = os.path.join(folder_dataset, 'testing.pkl')
    fn_result_emb = os.path.join(folder_dataset, 'embd_poly_'+appendix_this_run+'.embeddings')
    # pre-obtained clustering result of the graph on training data, the codes for this step could be found here: https://github.com/dakuang/symnmf
    fn_clustering = os.path.join(folder_dataset, 'snmf' + str(num_poly) + '.mat')

    # build graph
    training_data = sio.loadmat(fn_training)
    num_nodes = max(max(training_data['rows'][0])+1, max(training_data['cols'][0])+1)
    G = load_edgelist(training_data)

    # deepwalk
    dw = Poly_DeepWalk(num_nodes=num_nodes,
                       dim=dim,
                       num_poly=num_poly,
                       fn_clustering=fn_clustering, fn_testing=fn_testing, fn_result=fn_result_emb,
                       appendix_this_run=appendix_this_run,
                       appendix_path_this_run=appendix_path_this_run,
                       path_length=path_length,
                       window_size=window_size,
                       num_neg=num_neg)

    dw.train(G=G,
             num_walks_per_node=num_walks_per_node)