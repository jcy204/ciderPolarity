from .suggest_seeds import set_seeds, custom_seeds, seed_dict
from .stopwords import stoplist
from . import load_polarities
from . import create_embeddings
from . import run_bootstrapping
from . import create_vader
from . import utils_funcs
import warnings
import pickle
import os









class CIDER:

    def __init__(self,
                 fileinput,
                 output,
                 pos_seeds=None,
                 neg_seeds=None,
                 predefined_seeds='derived_twitter',
                 max_polarities_returned=5000,
                 preprocessing = 'default',
                 stopwords='nltk',
                 iterations=100,
                 verbose=True,
                 no_below=100,
                 keep=[]):
        '''
        Create domain specific lexicons.

        The code can be run as follows:
            cdr = CIDER(fileinput,output)
            cdr.fit_transform() 

        Parameters
        ----------
        fileinput - STRING/LIST - either list of STR or path of .json/.jsonl file using the format:
                                  {'text': example_text1}
                                  {'text': example_text2}
                                  {'text': example_text3}
                                    ... 
        output - STRING - path of folder for saving files
        pos_seeds - LIST/DICT - upper pole seed set. Either list, e.g. ['good','great'] 
                                                         or dict, e.g. {'good':1,'great':2}

        neg_seeds - LIST/DICT - lower pole seed set. Either list, e.g. ['bad','terrible'] 
                                                         or dict, e.g. {'bad':1,'terrible':2} (keep values +ve)

        predefined_seeds -STR - choose from 'derived_twitter', 'truney', 'finance', 'twitter', 'gender', 'historical'
        max_polarities_returned - INT - maximum number of words returned
        preprocessing - STR/FUNC/NONETYPE - set as 'default' for default preprocessing, None for text.split() or a 
                                            custom function taking one input (text) and returning a preprocessed and
                                            tokenised list of words/tokens.
        stopwords - STR/list/None - set as 'default' to remove nltk stopwords. Or add custom list/None. 
                                    ONLY WORKS WITH "preprocessing = 'default'"
        iterations - INT - the number of iterations for label propagation 

        verbose - BOOL - display progress
        no_below - INT - exclude words that occur less frequently than this
        keep - LIST/None - List of words to prevent being excluded
        '''
        if type(fileinput) == str:
            if not os.path.isfile(fileinput):
                raise FileNotFoundError('No file exists at the location specified')
        self.FILEINPUT = fileinput
        self.ITERATIONS = iterations
        if stopwords == 'default': stopwords = stoplist
        if stopwords == None: stopwords = []
        self.STOPWORDS = stopwords
        self.VERBOSE = verbose
        self.PREPROCESSING = preprocessing


        ## Seed Words
        self.POS_SEEDS = pos_seeds
        self.NEG_SEEDS = neg_seeds
        self.PREDEFINED_SEEDS = predefined_seeds
        self.SEEDS = set_seeds(self)

        ## Filtering Thresholds
        if keep == None: keep = []
        self.KEEP = sorted(set(self.clean_text(' '.join(keep))))
        self.NO_ABOVE_1 = 0.5
        self.NO_ABOVE_2 = 0.1
        self.NO_BELOW = no_below
        self.STATE = 0
        self.NN = 25
        self.LINES = 0
        self.MAX = max_polarities_returned

        ## Save Locations
        if not os.path.isdir(output):
            raise FileNotFoundError('No folder exists at the location specified')

        self.OUTPUT = output
        self.POLARITY_OUTPUT = f"{self.OUTPUT}polarities.json"

        self.paths = {
            'dict': f'{self.OUTPUT}dict.pkl',
            'ppmi': f'{self.OUTPUT}ppmi.pkl',
            'cooc': f'{self.OUTPUT}cooc.pkl',
            'ppmi_index': f'{self.OUTPUT}ppmi_index.pkl',
            'output_pols': f'{self.OUTPUT}output.json'
        }

        ## returned dataframe parameters
        self.CI = False
        self.STD = True
        self.return_all = False
        self.scale = True

        ## filtering polarities parameters. These can be adjusted.
        self.NEU_THRESH=0.45,
        self.VAR_UPPER=0.9,
        self.VAR_LOWER=0.5
        


    def generate_seeds(self, pos_initial, neg_initial, n=10, return_all=False, sentiment=True):
        '''
        Suggests Custom Seed words based off of PPMI relationships and frequency. These words are just
        suggestions and are not neccessarily the best for your data.
        
        Parameters
        ----------
        pos_initial - list - Small input list to base seeds off, e.g. ['amazing', 'brilliant', 'great']
        neg_initial - list - Small input list to base seeds off, e.g. ['terrible', 'rubbish', 'bad']
        n - INT - number of words returned in the produced pos & neg seed sets
        return_all - BOOL - set as True to return full filtered dataframe
        sentiment - BOOL - set as False if the generating non-sentiment seeds 
        '''
        return custom_seeds(self, pos_initial, neg_initial, n, return_all,sentiment)
    


    def fit(self, full_run=True, remove_neutral=True):
        '''
        Train CIDER and fit a VADER model on the outputs

        Parameters
        ----------
        full_run - BOOL - set as False to just return previously executed results
        remove_neutral - BOOL - set as True to remove words from VADER that CIDER 
                                classifies as neutral
        '''
        if full_run:
            ## Generating embeddings
            for ind, _ in enumerate(utils_funcs.text_iterate(self)):
                continue
            self.LINES = ind + 1
            if (self.LINES < 30000) & (self.NO_BELOW > 30):
                warnings.warn(f"\nThe size of the dataset ({self.LINES}) is relatively small. " 
                              "Consider reducing the 'no_below' parameter to improve returned " 
                              "polarities and to avoid potential errors.", stacklevel=2)

            create_embeddings.embed_text(self)

            ## Running Bootstrapping
            run_bootstrapping.propogate_labels(self)
        
        ## Loading Polarities
        self.create_df()
        self.train_VADER(remove_neutral)



    def transform(self, save_outputs=False, return_outputs = True):
        '''
        Apply fitted VADER model on the inputs.

        Parameters
        ----------
        save_outputs - BOOL - save the outputs into .json file
        return_outputs - BOOL - return the outputs as a list
        '''
        return create_vader.apply_vader(self, save_outputs, return_outputs)
    


    def fit_transform(self, save_outputs=False, return_outputs = True, full_run = True, remove_neutral=True):
        '''
        Train CIDER and fit a VADER model on the outputs. Apply fitted VADER model on the inputs
        
        Parameters
        ----------
        save_outputs - BOOL - save the outputs into .json file
        return_outputs - BOOL - return the outputs as a list
        full_run - BOOL - set as False to just return previously executed results
        remove_neutral - BOOL - set as True to remove words from VADER that CIDER 
                                classifies as neutral
        '''
        self.fit(full_run, remove_neutral)
        return self.transform(save_outputs, return_outputs)
    


    def create_df(self):
        'returns polarities only'
        self.polarities = load_polarities.make_df(self)



    def train_VADER(self,remove_neutral=True):
        '''
        trains VADER only        
        Parameters
        ----------
        remove_neutral - BOOL - set as True to remove words from VADER that CIDER 
                                classifies as neutral
        '''
        self.classify = create_vader.modify_vader(self,remove_neutral)
        

    def clean_text(self,text):
        '''
        cleans and tokenises the text
        '''
        if self.PREPROCESSING == 'default':
            return utils_funcs.default_clean_text(self, text)
        if self.PREPROCESSING == None:
            return text.split()
        if callable(self.PREPROCESSING):
            return self.PREPROCESSING(text)

    def save(self, fname, var):
        'Function used to pickle (save) variables'
        with open(fname, "wb") as f:
            pickle.dump(var, f)



    def load(self, fname):
        'Loads pickled (saved) variables'
        if fname.lower() in self.paths:
            fname = self.paths[fname.lower()]

        with open(fname, "rb") as f:
            return pickle.load(f)


    
    def get_seeds(self,input = None):
        '''
        Either returns set seeds, or if input != None, the corresponding seed set for that input is returned 
        '''
        if input:
            return seed_dict[input.lower()]
        else:
            return self.SEEDS
