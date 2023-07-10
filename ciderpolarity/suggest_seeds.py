import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
r_arg = {'pct':True,'method':'dense'}

"""
Seed words for propagating polarity scores.
"""
POSITIVE_DERIVED_TWITTER = [
    'lovely', 'excellent', 'fortunate', 'pleasant', 'delightful',
    'perfect', 'loved', 'love', 'good', 'nice', 'beautiful', 'great',
    'enjoy', 'gorgeous', 'awesome', 'amazing', 'excited','loves','♥','🙂'
]

NEGATIVE_DERIVED_TWITTER = [
    'bad', 'horrible', 'hate', 'damn', '🙁', 'shit', 'shitty', 'fuck', 'hell',
    'wtf', 'hated','stupid', 'terrible', 'nasty', 'awful', 'worst', 'crap','crappy',
    'sad','bitch','hates'
]

POSITIVE_TURNEY = [ "good", "nice", "excellent", "positive", "fortunate", "correct", "superior"]
NEGATIVE_TURNEY = [ "bad","terrible","poor","negative","unfortunate","wrong","inferior"]

POSITIVE_FINANCE = ["successful","excellent","profit","beneficial","improving","improved","success","gains","positive"]
NEGATIVE_FINANCE = ["negligent","loss","volatile","wrong","losses","damages","bad","litigation","failure","down","negative"]

POSITIVE_TWEET = [ "love","loved","loves","awesome","nice","amazing","best","fantastic","correct","happy"]
NEGATIVE_TWEET = [ "hate","hated","hates","terrible","nasty","awful","worst","horrible","wrong","sad"]

MALE = ["man", "male", "boy", "gentleman", "mr", "masculine", "dad", "father", "brother", "son"]
FEMALE = ["female", "woman", "girl", "lady", "mom", "mum", "sister", "mother", "feminine", "daughter", "ms", "mrs", "miss"]

POSITIVE_HIST = [ "good","lovely","excellent","fortunate","pleasant","delightful","perfect","loved","love","happy"]
NEGATIVE_HIST = [ "bad","horrible","poor","unfortunate","unpleasant","disgusting","evil","hated","hate","unhappy"]

seed_dict = {'derived_twitter': [POSITIVE_DERIVED_TWITTER, NEGATIVE_DERIVED_TWITTER],
    'historical': [POSITIVE_HIST, NEGATIVE_HIST],
    'finance': [POSITIVE_FINANCE, NEGATIVE_FINANCE],
    'twitter': [POSITIVE_TWEET, NEGATIVE_TWEET],
    'gender': [MALE, FEMALE],
    'truney': [POSITIVE_TURNEY, NEGATIVE_TURNEY],
    
}


    
def set_seeds(SS):
    'returns seed options, or filters given seeds'
    
    try:
        if SS.POS_SEEDS:
            seeds = [SS.POS_SEEDS, SS.NEG_SEEDS]
        elif SS.PREDEFINED_SEEDS:
            seeds = seed_dict[SS.PREDEFINED_SEEDS.lower()]
            

        if type(seeds[0]) == list:
            pos_seeds = [SS.clean_text(i)[0] for i in sorted(set(seeds[0]))]
            neg_seeds = [SS.clean_text(i)[0] for i in sorted(set(seeds[1]))]
            return [pos_seeds,neg_seeds]

        if type(seeds[0]) == dict:
            pos_seeds = {SS.clean_text(i)[0]: seeds[0][i] for i in sorted(seeds[0])}
            neg_seeds = {SS.clean_text(i)[0]: seeds[1][i] for i in sorted(seeds[1])}
            return [pos_seeds,neg_seeds]
    except:
        valid_options = ', '.join(list(seed_dict))
        raise ValueError(f"""\nInvalid seed type. Please use ONE of the following options: {valid_options}.
                         Or supply your own in the following format: [upper_list_of_seeds, lower_list_of_seeds]""")



def custom_seeds(SS, pos_initial, neg_initial, n=10, return_all=False, sentiment = True):
    try:
        SS.load('PPMI')
    except:
        SS.gen_only()
        
    gdict = SS.load('dict')
    token2id = gdict.token2id    
    ppmi = SS.load('ppmi')
    
    pos_loc = [token2id[i] for i in pos_initial if i in token2id]
    neg_loc = [token2id[i] for i in neg_initial if i in token2id]

    pos_score = ppmi[pos_loc].toarray().mean(axis=0)
    neg_score  = ppmi[neg_loc].toarray().mean(axis=0)
    df_ppmi = pd.DataFrame({'upper_seeds':pos_score, 'lower_seeds':neg_score},index = token2id)
    df_ppmi['counts'] = [gdict.cfs[token2id[i]] for i in token2id]


    if sentiment:
        SIA = SentimentIntensityAnalyzer()
        df_ppmi['Vader'] = [SIA.lexicon[x] if x in SIA.lexicon else 0 for x in token2id]
        df_ppmi = df_ppmi[df_ppmi['Vader']!=0].copy()
        df_ppmi['r_diff'] = df_ppmi.rank(**r_arg).diff(axis=1)['lower_seeds']*-1
        df_ppmi['compound_seed_metric'] = (df_ppmi.counts*df_ppmi.Vader).rank(**r_arg)*0.5
        df_ppmi['compound_seed_metric'] += df_ppmi.r_diff.rank(**r_arg)
    else:
        df_ppmi['r_diff'] = df_ppmi.rank(**r_arg).diff(axis=1)['lower_seeds']*-1
        df_ppmi['compound_seed_metric'] = (df_ppmi.counts*df_ppmi.r_diff/df_ppmi.r_diff.abs()).rank(**r_arg)*0.5 
        df_ppmi['compound_seed_metric'] += df_ppmi.r_diff.rank(**r_arg)

    df_ppmi['compound_seed_metric'] = (df_ppmi['compound_seed_metric']*2/1.5) - 1
    df_ppmi = df_ppmi.sort_values('compound_seed_metric')
    
    if return_all:
        return df_ppmi
    
    df_ppmi = df_ppmi[~df_ppmi.index.isin(pos_initial+neg_initial)]
    pos_return = df_ppmi.tail(n).index.tolist()
    neg_return = df_ppmi.head(n).index.tolist()
    
    return [pos_return, neg_return]
    
