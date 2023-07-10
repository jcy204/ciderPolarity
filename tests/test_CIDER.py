import sys
sys.path.insert(1, "../CIDER/")

from CIDER import CIDER
import csv
import os
## Seed Types
POS = ['lovely', 'excellent', 'fortunate', 'excited', 'loves', '♥', '🙂']
NEG = ['bad', 'horrible', 'hate', 'crappy', 'sad', 'bitch', 'hates']

POS_d = {'lovely':1, 'excellent':2, 'fortunate':4, 'excited':1, 'loves':2, '♥':1, '🙂':2}
NEG_d = {'bad':1, 'horrible':2, 'hate':4, 'crappy':1, 'sad':2, 'bitch':1, 'hates':2}

## File Paths
base_path = os.getcwd()
path = base_path+'/test_data.csv'
output = base_path+'/test_outputs/'

## Load Text
texts = []
with open(path) as p:
    reader = csv.reader(p)
    for r in reader:
        texts.append(' ,'.join(r))


## Initialise CIDER with different parameters.


cdr_Test_1 = CIDER(texts,
                   output,
                   iterations=150,
                   stopwords='nltk',
                   preprocessing = None,
                   keep=['one', 'two', 'three', 'xyzxyzabc'],
                   max_polarities_returned=1000,
                   predefined_seeds='derived_twitter',
                   verbose=False)

cdr_Test_2 = CIDER(path,
                   output,
                   iterations=10,
                   stopwords=None,
                   preprocessing = 'default',
                   no_below=50,
                   keep=None,
                   max_polarities_returned=500,
                   predefined_seeds='gender',
                   verbose=False)

def custom_preprocess(text):
    text = [i[::-1] for i in text.upper().split()]
    return text

cdr_Test_3 = CIDER(texts,
                   output,
                   stopwords=[],
                   preprocessing = custom_preprocess,
                   keep=['one', 'two', 'three', 'xyzxyzabc'],
                   max_polarities_returned=100,
                   pos_seeds=POS,
                   neg_seeds=NEG,
                   verbose=False)


cdr_Test_4 = CIDER(path,
                  output,
                  iterations=100,
                  stopwords=['i', 'test', 'code', 'functionality'],
                  keep=[],
                  no_below=15,
                  max_polarities_returned=1000,
                  pos_seeds=POS_d,
                  neg_seeds=NEG_d,
                  verbose=False)

cdr_Test_5 = CIDER(texts, output)


## Run CIDER
Test_1_OUT = cdr_Test_1.fit_transform(save_outputs=True, return_outputs=True)
Test_2_OUT = cdr_Test_2.fit_transform(save_outputs=True, return_outputs = False)
Test_3_OUT = cdr_Test_3.fit_transform(save_outputs=False, return_outputs=False)
Test_4_OUT = cdr_Test_4.fit_transform(return_outputs=True)
cdr_Test_5.fit_transform(return_outputs = False)
Test_5_OUT = cdr_Test_5.fit_transform(full_run=False)

assert type(Test_1_OUT) == list; print('Test_1: Complete')
assert Test_2_OUT == None; print('Test_2: Complete')
assert Test_3_OUT == None; print('Test_3: Complete')
assert type(Test_4_OUT) == list; print('Test_4: Complete')
assert type(Test_5_OUT) == list; print('Test_5: Complete')

## Test Clean Function
text = 'Test of the ClEaN FuNctION 🔥'
assert cdr_Test_1.clean_text(text) == ['Test', 'of', 'the', 'ClEaN', 'FuNctION', '🔥'], 'text cleaning broken'
assert cdr_Test_2.clean_text(text) == ['test', 'of', 'the', 'clean', 'function', '🔥'], 'text cleaning broken'
assert cdr_Test_3.clean_text(text) == ['TSET', 'FO', 'EHT', 'NAELC', 'NOITCNUF', '🔥'], 'text cleaning broken'
assert cdr_Test_4.clean_text(text) == ['of', 'the', 'clean', 'function', '🔥'], 'text cleaning broken'


## Test Seed Word Generation
Pos, Neg = cdr_Test_1.generate_seeds(['good','brilliant','love'],['bad','terrible','hate'],n=20, sentiment = True)
df = cdr_Test_1.generate_seeds(['she','her','hers','girl'],['he','him','his','boy'], sentiment = False,return_all=True)