from sklearn import preprocessing
from collections import Counter
from scipy import sparse
import numpy as np
try:
    from tqdm.auto import tqdm
except:
    from tqdm import tqdm
import json


def propogate_labels(CDR):

    ### Loading words to keep
    keep = list(CDR.KEEP)
    pos_seeds, neg_seeds = CDR.SEEDS
    keeptokens = list(pos_seeds)+ list(neg_seeds) + keep

    ### Filtering gdict
    gdict = CDR._load(fname = CDR.OUTPUT + 'dict.pkl')
    vocab = gdict.token2id
    gdict.filter_extremes(no_above=CDR.NO_ABOVE_2, no_below=CDR.NO_BELOW, keep_tokens = keeptokens)

    ## Filtering out CDR.MAX tokens
    keeptoken_ids = [gdict.token2id[i] for i in keeptokens if i in gdict.token2id] 
    to_keep_init = [i[0] for i in Counter(gdict.dfs).most_common(CDR.MAX)] # At most CDR.MAX words
    to_keep = list(dict.fromkeys(keeptoken_ids+ to_keep_init))[:CDR.MAX]
    to_keep = sorted(to_keep, key=lambda w: gdict.dfs[w], reverse=True)

    gdict.filter_tokens(good_ids=to_keep) # Remove Bad Tokens
    u = np.load(CDR.OUTPUT + "vec-u.npy")

    sub_vec_m = preprocessing.normalize(u, copy=False)

    word_list = set(gdict.token2id.keys()).union(keeptokens)
    word_list =  set([w for w in word_list if w in vocab])

    keep_indices = [vocab[word] for word in word_list]
    sub_vec_m = sub_vec_m[keep_indices, :]

    params = {'POLARITY_OUTPUT': CDR.POLARITY_OUTPUT,
              'ITERATIONS': CDR.ITERATIONS,
              'VERBOSE': CDR.VERBOSE,
              'STATE': CDR.STATE,
              'SEEDS': CDR.SEEDS}


    ### Transition Matrix
    trans_M = transition_matrix(sub_vec_m, CDR.NN)

    polarity_bootstrap(trans_M, word_list, params)


def transition_matrix(embeddings,nn):
    """
    Build a probabilistic transition matrix from word embeddings.
    """
    L = similarity_matrix(embeddings,nn)

    Dinv = np.diag([1.0 / L[i].sum() for i in range(L.shape[0])])
    L = L.dot(Dinv)

    return L


def similarity_matrix(embeddings, nn, arccos=False, similarity_power=1):
    """
    Constructs a similarity matrix from embeddings.
    nn argument controls the degree.
    """
    
    def make_knn(vec, nn=nn):
        vec[vec < vec[np.argsort(vec)[-nn]]] = 0
        return vec

    L = embeddings.dot(embeddings.T)
    if sparse.issparse(L):
        L = L.todense()
    if arccos:
        L = np.arccos(np.clip(-L, -1, 1)) / np.pi
    else:
        L += 1
        
    np.fill_diagonal(L, 0)
    L = np.apply_along_axis(make_knn, 1, L, nn = nn)
    
    return L ** similarity_power

def polarity_bootstrap(M, wordlist,params):
    iterations = params['ITERATIONS']
    verbose = params['VERBOSE']
    state = params['STATE']
    saveloc = params['POLARITY_OUTPUT']
    
    np.random.seed(state)
    
    with open(saveloc,'w') as newfile:
        for _ in tqdm(range(iterations), disable=(not verbose)):
            [rpos, rneg] = run_random_walks(M, wordlist, params)

            pols = {w: [rpos[i], rneg[i]] for i, w in enumerate(wordlist)}
            json.dump(pols, newfile)
            newfile.write('\n')
            


def run_random_walks(M, wordlist, params):

    positive_seeds, negative_seeds = params['SEEDS']

    
    pos_len = len(positive_seeds) - 2
    neg_len = len(negative_seeds) - 2
    
    if pos_len > 10:  pos_len = int(len(positive_seeds)*0.8)
    if neg_len > 10:  neg_len = int(len(negative_seeds)*0.8)
    
    if pos_len <1: pos_len=1
    if neg_len <1: neg_len=1

    pos_seeds = np.random.choice(sorted(positive_seeds), pos_len, replace=False)
    neg_seeds = np.random.choice(sorted(negative_seeds), neg_len, replace=False)        
    
    if type(positive_seeds) == dict:
        pos_seeds = {i:positive_seeds[i] for i in pos_seeds}
        neg_seeds = {i:negative_seeds[i] for i in neg_seeds}
    
    
    polarities = random_walk(M, wordlist, pos_seeds, neg_seeds, beta=0.9)
    
    return polarities




def random_walk(M, words, positive_seeds, negative_seeds, beta=0.9):
    """
    Learns polarity scores via random walks with teleporation to seed sets.
    Main method used in paper. 
    """
        
    def run_random_walk(M, teleport, beta):
        def update_seeds(r):
            r += (1 - beta) * teleport / (np.sum(teleport)+1)

        return run_iterative(
            M * beta, np.ones(M.shape[1]) / M.shape[1], update_seeds)

    if not type(positive_seeds) is dict:
        positive_seeds = {word: 1.0 for word in positive_seeds}
        negative_seeds = {word: 1.0 for word in negative_seeds}
    
    

    rpos = run_random_walk(
        M, initialize_weighted_set(words, positive_seeds), beta)
    rneg = run_random_walk(
        M, initialize_weighted_set(words, negative_seeds), beta)
    

        
    return [rpos, rneg]

### HELPER METHODS #####
def initialize_weighted_set(words, seed_weights):
    return np.array(
        [seed_weights[word] if word in seed_weights else 0.0 for word in words]
    )


def run_iterative(M, r, update_seeds, max_iter=50, epsilon=1e-6):
    for _ in range(max_iter):
        last_r = np.array(r)
        r = np.dot(M, r)
        update_seeds(r)

        if np.abs(r - last_r).sum() < epsilon:
            break
    return r




