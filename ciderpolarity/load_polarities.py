import pandas as pd
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import scipy.stats as st
import numpy as np
import json
from collections import Counter
from collections import defaultdict


def make_df(SS):
    '''
    Makes df using data
    '''
    
    ### Collecting the Data
    pols = {'pos':defaultdict(list),'neg':defaultdict(list)}
    
    with open(SS.POLARITY_OUTPUT) as file:
        for row in file:
            boot_pols = json.loads(row)
            for w in boot_pols:
                if np.isnan(boot_pols[w][0]):
                    continue
                if np.isnan(boot_pols[w][1]):
                    continue
                pols['pos'][w].append(boot_pols[w][0])
                pols['neg'][w].append(boot_pols[w][1])                
                
    df = pd.DataFrame(pols)
    df['polarity'] = df.pos.apply(np.array)/(df.pos.apply(np.array)+df.neg.apply(np.array))
        
    ## Word Counts
    index = SS.load( fname = SS.OUTPUT + 'dict.pkl')
    word_freq = Counter({i[0]:i[1] for i in index.most_common()})
    df['freq'] = df.index.map(word_freq)     
        
    if SS.CI:
        df[f'CI_{SS.CI}'] = df['polarity'].apply(lambda x: st.norm.interval(confidence=0.95,scale=st.sem(x))[1])
        
    if SS.STD:
        df['polarity_std'] = df['polarity'].apply(np.std)
        
    
    ### Vader
    SIA = SentimentIntensityAnalyzer()
    df['VADER'] = df.index.to_series().apply(lambda x: SIA.lexicon[x.lower()] if x in SIA.lexicon else np.nan)


    if SS.return_all:
        return df
    
    df['pos_prox'] = df.pos.apply(np.mean)
    df['neg_prox'] = df.neg.apply(np.mean)
    df['polarity'] = df.polarity.apply(np.mean)
    df = df.drop(['pos','neg'],axis=1)
    
    ### Scaling the Data
    scaler = StandardScaler(with_std=False)
    df["polarity"] = scaler.fit_transform(df["polarity"].values.reshape(-1, 1))
        
    scale = df.polarity.abs().max()
    df['polarity'] = 4*df.polarity/scale
    df = df[['polarity','VADER','polarity_std','pos_prox','neg_prox','freq']]

    return df.sort_values('polarity')
