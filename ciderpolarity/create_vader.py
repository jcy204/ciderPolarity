from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from .utils_funcs import text_iterate


SIA = SentimentIntensityAnalyzer()
r_arg = {'pct':True,'method':'dense'}

################################ CHECK THESE THRESHOLDS ####################################
def modify_vader(SS):
    try: 
        df = SS.polarities
    except:
        SS.create_df()
        df = SS.polarities
    
    ## Filter DF
    df_pos, df_neg, remove = filter_df(df, SS.NEU_THRESH, SS.VAR_UPPER, SS.VAR_LOWER)
    return make_VADER_custom(df_pos, df_neg, remove)

    
def make_VADER_custom(positive, negative, remove):
    SIA_Custom = SentimentIntensityAnalyzer()
    
    for i in remove: 
        try: del SIA_Custom.lexicon[i]
        except: pass
    
    for i in positive.itertuples():
        SIA_Custom.lexicon[i.Index] = i.polarity
        if i.Index in SIA_Custom.emojis:
            del SIA_Custom.emojis[i.Index]

    for i in negative.itertuples():
        SIA_Custom.lexicon[i.Index] = i.polarity
        if i.Index in SIA_Custom.emojis:
            del SIA_Custom.emojis[i.Index]
            
    return SIA_Custom

def filter_df(df,neu_thresh, var_upper, var_lower):

    df_remove = df[ (df.pos_prox.rank(**r_arg) < neu_thresh) 
                     & (df.neg_prox.rank(**r_arg) < neu_thresh)].index.tolist()
    
    df_k = df[~df.index.isin(df_remove)].copy()

    df_pos = df_k[(df_k.pos_prox.rank(**r_arg)>var_upper) & (df_k.neg_prox.rank(**r_arg)<var_lower)].copy()
    df_neg = df_k[(df_k.neg_prox.rank(**r_arg)>var_upper) & (df_k.pos_prox.rank(**r_arg)<var_lower)].copy()

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
