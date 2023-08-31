from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer, SentiText, normalize, BOOSTER_DICT
import json
import numpy as np
from .utils_funcs import text_iterate


SIA = SentimentIntensityAnalyzer()
r_arg = {'pct':True,'method':'dense'}

################################ CHECK THESE THRESHOLDS ####################################
def modify_vader(CDR,remove_neutral=True):
    '''
    remove_neutral - BOOL - set as True to remove words from VADER that CIDER 
                            classifies as neutral   
    '''

    try: 
        df = CDR.polarities
    except:
        CDR.create_df()
        df = CDR.polarities
    
    ## Filter DF
    df_pos, df_neg, remove = filter_df(df, CDR.NEU_THRESH, CDR.POL_THRESH)
    if remove_neutral == False:
        remove = []
    return make_VADER_custom(df_pos, df_neg, remove, CDR.SENTIMENT)


    
def make_VADER_custom(positive, negative, remove, sentiment):
    SIA_Custom = SentimentIntensityAnalyzer()
    
    for i in remove: 
        try: del SIA_Custom.lexicon[i]
        except: pass
    if sentiment == False:
        SIA_Custom.lexicon = {}

    for i in positive.itertuples():
        SIA_Custom.lexicon[i.Index] = i.polarity
        if i.Index in SIA_Custom.emojis:
            del SIA_Custom.emojis[i.Index]

    for i in negative.itertuples():
        SIA_Custom.lexicon[i.Index] = i.polarity
        if i.Index in SIA_Custom.emojis:
            del SIA_Custom.emojis[i.Index]
            
    return SIA_Custom

def filter_df(df,neu_thresh, pol_thresh):

    df_remove = df[ (df.pos_prox.rank(**r_arg) < neu_thresh) 
                     & (df.neg_prox.rank(**r_arg) < neu_thresh)].index.tolist()
    
    df_k = df[~df.index.isin(df_remove)].copy()

    df_k['metric'] = df_k.pos_prox.rank(**r_arg) - df_k.neg_prox.rank(**r_arg)
    df_k['metric'] = (df_k.metric-df_k.metric.min())/(df_k.metric.max()-df_k.metric.min())

    df_pos = df_k[df_k.metric > 0.5+pol_thresh/2].copy()
    df_neg = df_k[df_k.metric < 0.5-pol_thresh/2].copy()

    return df_pos, df_neg, df_remove


def apply_vader(self,save_outputs,return_outputs):
    if not self.classify:
        raise ValueError(f"""Apply model.fit() before model.transform()""")
    
    results = []
    
    if save_outputs:
        if self.VERBOSE: print(f"Saving Classified Text to: {self.paths['output_pols']}")
        with open(self.paths['output_pols'],'w') as newfile:
            for row in text_iterate(self, show=self.VERBOSE):
                result = [row, self.classify.polarity_scores(row)]
                output = json.dumps({'body':result[0],'polarity':result[1]})
                newfile.write(output+'\n')
                
                if return_outputs:
                    results.append(result)

    elif return_outputs:
        if self.VERBOSE: print('Returning Classified Text')
        for row in text_iterate(self, show=self.VERBOSE):
            result = [row, self.classify.polarity_scores(row)]
            results.append(result)

    if return_outputs:
        return results
    

def intensity(cdr, text):
    """
    Return a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative
    valence.
    """
    

    # convert emojis to their textual descriptions
    text_no_emoji = ""
    prev_space = True
    for chr in text:
        if chr in cdr.emojis:
            # get the textual description
            description = cdr.emojis[chr]
            if not prev_space:
                text_no_emoji += ' '
            text_no_emoji += description
            prev_space = False
        else:
            text_no_emoji += chr
            prev_space = chr == ' '
    text = text_no_emoji.strip()

    sentitext = SentiText(text)

    sentiments = []
    words_and_emoticons = sentitext.words_and_emoticons
    for i, item in enumerate(words_and_emoticons):
        valence = 0
        # check for vader_lexicon words that may be used as modifiers or negations
        if item.lower() in BOOSTER_DICT:
            sentiments.append(valence)
            continue
        if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
                words_and_emoticons[i + 1].lower() == "of"):
            sentiments.append(valence)
            continue

        sentiments = cdr.sentiment_valence(valence, sentitext, item, i, sentiments)

    sentiments = cdr._but_check(words_and_emoticons, sentiments)
    if sentiments:
        sum_s = float(sum(np.abs(sentiments)))
        punct_emph_amplifier = SIA._punctuation_emphasis(text)
        if sum_s > 0:
            sum_s += punct_emph_amplifier
        elif sum_s < 0:
            sum_s -= punct_emph_amplifier


        compound = normalize(sum_s)
    
    return compound
