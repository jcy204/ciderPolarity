from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer, SentiText, normalize, BOOSTER_DICT
import json
import numpy as np
from .utils_funcs import text_iterate
import math

SIA = SentimentIntensityAnalyzer()
r_arg = {'pct':True,'method':'dense'}


################################ CHECK THESE THRESHOLDS ####################################
def modify_vader(CDR,remove_neutral=True):
    '''
    remove_neutral - BOOL - set as True to remove words from VADER that CIDER 
                            classifies as neutral   
    '''
    try: 
        CDR.polarities
    except:
        CDR.create_df()    
        
    return Custom_VADER(CDR, remove_neutral)

    


    
######## Rewritten VADER Functions
class Custom_VADER(SentimentIntensityAnalyzer):
    
    def __init__(self, cdr, remove_neutral):
        super().__init__()
        self.REMOVE_NEUTRAL = remove_neutral
        self.edit_lexicon(cdr)

    def edit_lexicon(self, cdr):
        df = cdr.polarities.copy()
        
        ## Filtering DF
        df_remove = df[ (df.pos_prox.rank(**r_arg) < cdr.NEU_THRESH) 
                      & (df.neg_prox.rank(**r_arg) < cdr.NEU_THRESH)].index.tolist()

        df_k = df[~df.index.isin(df_remove)].copy()

        df_k['metric'] = (df_k.pos_prox - df_k.neg_prox)#.rank(**r_arg)
        
        keep = int(len(df_k)*cdr.POL_THRESH/2)
        df_pos = df_k.sort_values('metric').tail(keep)
        df_neg = df_k.sort_values('metric').head(keep)

        ## editing lexicon
        if self.REMOVE_NEUTRAL == False:
            df_remove = []
    
        for i in df_remove: 
            try: del self.lexicon[i]
            except: pass
            
        if cdr.SENTIMENT == False:
            self.lexicon = {}

        for i in df_pos.itertuples():
            self.lexicon[i.Index] = i.polarity
            if i.Index in self.emojis:
                del self.emojis[i.Index]

        for i in df_neg.itertuples():
            self.lexicon[i.Index] = i.polarity
            if i.Index in self.emojis:
                del self.emojis[i.Index]
    
    
    def score_valence(self, sentiments, text):

        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)

            # compute intensity from text
            sum_c = float(sum(np.abs(sentiments)))
            sum_c += punct_emph_amplifier
            intensity = normalize(sum_c)

            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg": round(neg, 3),
             "neu": round(neu, 3),
             "pos": round(pos, 3),
             "compound": round(compound, 4),
             "intensity": round(intensity,4)}

        return sentiment_dict
