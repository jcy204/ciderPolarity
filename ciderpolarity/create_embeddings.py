import numpy as np
from collections import Counter
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds
from scipy.sparse import diags, lil_matrix
from gensim.matutils import corpus2csc
from gensim.corpora import Dictionary
from tqdm.auto import tqdm
from utils_funcs import text_iterate



def embed_text(SS):

    ### Load Tweets
    gdict = Dictionary()
    for row in text_iterate(SS, show=SS.VERBOSE):
        
        processed_comment = SS.clean_text(row)
        gdict.add_documents([processed_comment],prune_at = None)

    ### Filter Dictionary
    pos_seeds, neg_seeds = SS.SEEDS
    keep = list(SS.KEEP)

    keeptokens = list(pos_seeds)+ list(neg_seeds) + keep
    gdict.filter_extremes(no_above=SS.NO_ABOVE_1, no_below=SS.NO_BELOW, keep_tokens = keeptokens)
    gdict.compactify()

    ### Coocurrence
    if SS.VERBOSE: print('Running Cooc')
    cooc = gen_cooc(gdict, SS)
    
    SS.save( fname = SS.OUTPUT + 'cooc.pkl', var = cooc)

    ### PPMI   
    if SS.VERBOSE: print('Running PPMI')
    ppmi = gen_PPMI(cooc,SS)

    SS.save( fname = SS.OUTPUT + 'ppmi.pkl', var = ppmi)
    SS.save( fname = SS.OUTPUT + 'ppmi_index.pkl', var = gdict.token2id)

    ### SVD
    if SS.VERBOSE: print('Making Low Dim')
    u,s, v = run_lowdim(ppmi,SS)


    np.save(SS.OUTPUT + "vec-u.npy", u)
    np.save(SS.OUTPUT + "vec-v.npy", v)
    np.save(SS.OUTPUT + "vec-s.npy", s)
    
    SS.save( fname = SS.OUTPUT + 'dict.pkl', var = gdict)

def gen_cooc(gdict,SS, tfidf = False):
    bow_corpus = []
    dup_corpus = Counter()


    for row in text_iterate(SS, show=SS.VERBOSE):
        processed_comment = SS.clean_text(row)
        mat = gdict.doc2bow(processed_comment)
        bow_corpus += [mat]
        dup_corpus.update(dict(mat)) # To subtract diagonal 

    if tfidf:
        return bow_corpus
        
    term_doc_mat = corpus2csc(bow_corpus)
    cooc = np.dot(term_doc_mat, term_doc_mat.T)
    diag = diags([dup_corpus[i] for i in sorted(dup_corpus)])

    cooc -= diag
    return cooc

def gen_PPMI(sparse_matrix, SS, cds = True):
    total_sum = sparse_matrix.sum()
    row_sum = np.array(sparse_matrix.sum(axis=1))
    col_sum = np.array(sparse_matrix.sum(axis=0))
    if cds:
        col_sum = col_sum**(0.75)

    row_sum = row_sum / row_sum.sum()
    col_sum = col_sum / col_sum.sum()
    numer_mat = (sparse_matrix / total_sum)
    pmi = lil_matrix(numer_mat.shape)


    for ind, numer in tqdm(enumerate(numer_mat.T), total=len(row_sum), disable=(not SS.VERBOSE)):
        numer = numer.toarray()
        denom = np.multiply(row_sum[ind], col_sum.copy())
        denom[numer == 0] = 1
        ppmi_row = numer / denom
        ppmi_row[ppmi_row == 0] = 1

        pmi[ind, :] = np.maximum(np.log(ppmi_row), np.zeros(len(numer)))

    return pmi.tocsr()



def run_lowdim(ppmi,SS, dim=300):
    try:
        u, s, v = svds(ppmi,k=dim, maxiter = 5, solver ='arpack', random_state=0)
    except:
        u, s, v = randomized_svd(ppmi, n_components=dim, n_iter=5,random_state=None)
        if SS.VERBOSE: print('RandomSVD')
    return u,s, v


    

