# CIDER Python Package

CIDER (Context Informed Dictionary and sEntiment Reasoner) is a Python library used to improve domain-specific sentiment analysis.

It generates, filters, and substitutes polarities into [VADER](https://github.com/cjhutto/vaderSentiment/). 
The approach taken to generate polarities is taken from [SocialSent](https://github.com/williamleif/socialsent).

## Contents

- [Installation](#installation)
- [Overview](#overview)
- [Examples](#examples)


## Installation

Before you begin, ensure you have met the following requirements:

* You have installed Python 3.7 or later.
* You have a Windows/Linux/Mac machine.

To install CIDER, follow these steps:

```bash
pip install ciderpolarity
```

## Overview

The easiest way to use the package is as follows:

```python
from ciderpolarity import CIDER

# For a running example, the ideal input will have many thousands of lines.
texts = ['Really hate this heat. Just want AC',
         'I love an icecream in this heat!',
         'I’m melting - terrible weather!',
         'Very dehydrated in this heat',
                ...                    ,
         'this sunny weather is great',
         'Oh my icecream is melting',
         'My AC is broken! 🥵'],


output_folder = '/path/to/output/folder/'

cdr = CIDER(input_file, output_folder)
results = cdr.fit_transform()
```

This trains the model, creating a customised VADER classifier, before classifying the provided input using the model. An example output is as follows:

```python
results = [
    ['Really hate this heat. Just want AC', {"neg":0.6, "neu":0.4, "pos":0.0, "compound":-0.6}],
    ['I love an icecream in this heat!', {"neg":0.0, "neu":0.5, "pos":0.5, "compound":0.6}],
    ['I’m melting - terrible weather!', {"neg":0.7, "neu":0.3, "pos":0.0, "compound":-0.7}],
    ['Very dehydrated in this heat', {"neg":0.5, "neu":0.4, "pos":0.0, "compound":-0.5}],
            ...
    ['this sunny weather is great', {"neg":0.0, "neu":0.2, "pos":0.8, "compound":0.7}],
    ['Oh my icecream is melting', {"neg":0.3, "neu":0.4, "pos":0.3, "compound":0.0}],
    ['My AC is broken! 🥵', {"neg":0.6, "neu":0.4, "pos":0.0, "compound":-0.6}],
]

```

## Examples
**Some alternative ways to use the library are as follows:**

Applying CIDER to a saved dataset, adding custom seed words, custom stopwords, and tuning various parameters:

```python
POS_seeds = {'lovely':1, 'excellent':2, 'fortunate':4, 'excited':1, 'loves':2, '♥':1, '🙂':2}
NEG_seeds = {'bad':1, 'horrible':2, 'hate':4, 'crappy':1, 'sad':2, 'bitch':1, 'hates':2}

input_file = '/path/to/input/file.csv'
output = '/path/to/output/test_outputs/'

cdr_example = CIDER(input_file,                   # input path (one column csv file where each row is a text entry)
                    output,                       # output path
                    iterations=100,               # number of iterations for bootstrapped label propagation
                    stopwords=['i', 'it', 'the'], # custom stopwords, alternativly set as 'default' for the nltk set
                    keep=['code', 'python'],      # words to force into the final lexicon
                    no_below=5,                   # exclude words that occur fewer times than this
                    max_polarities_returned=3000, # maximum number of words returned
                    pos_seeds=POS_seeds,          # positive seeds with custom weighting
                    neg_seeds=NEG_seeds,          # negative seeds with custom weighting
                    verbose=False)                # whether to print progress or not
```

If the model only requires training, the following can be executed:

```python
cdr_example.fit()
```

And the resulting polarities (before filtering and scaling) can be viewed:

<img src="https://github.com/jcy204/ciderPolarity/blob/main/cdr_out_example.png?raw=true" alt="drawing" width="500"/>

___

### Generating Seedwords

Whilst CIDER has built in seed words (found [here](CIDER/suggest_seeds.py)), custom seed words can be generated and suggested. The following shows how this is carried out:

```python
Pos, Neg = cdr_example.generate_seeds(['good','brilliant','love'],['bad','terrible','hate'], n=20, sentiment = True)
```
Which looks at strongly polarised words which occur both often, are close to one seed set, and distant from the opposing seed set.

The following returns all words in the data, alongside their seed word suitability.
```python
df = cdr_example.generate_seeds(['good','brilliant','love'],['bad','terrible','hate'], return_all = True, sentiment = True)
```
