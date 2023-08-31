try:
    from tqdm.auto import tqdm
except:
    from tqdm import tqdm
import csv

from string import punctuation
translator = str.maketrans('', '', punctuation)


def default_clean_text(CDR, doc):
    "Input doc and return clean list of tokens"
    doc = doc.replace('\r', ' ').replace('\n', ' ')
    lower = doc.lower()  # all lower case
    nopunc = lower.translate(translator)  # remove punctuation
    words = nopunc.split()  # split into tokens
    if CDR.STOPWORDS:
        words = [
            w for w in words
            if w not in CDR.STOPWORDS and not w.isdigit()
        ]  # remove stopwords
    return words


def text_iterate(CDR, show=False, full=False):
    'Allows iteration over a list of text or a csc file'
    if isinstance(CDR.FILEINPUT, list):
        for row in tqdm(CDR.FILEINPUT, total=CDR.LINES,
                        disable=not show):
            yield row
    else:
        with open(CDR.FILEINPUT, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in tqdm(reader, total=CDR.LINES, disable=not show):
                if full:
                    yield row
                else:
                    yield ' ,'.join(row)


NEGATE = {"ain't",
 'aint',
 "aren't",
 'arent',
 "can't",
 'cannot',
 'cant',
 "couldn't",
 'couldnt',
 "daren't",
 'darent',
 'despite',
 "didn't",
 'didnt',
 "doesn't",
 'doesnt',
 "don't",
 'dont',
 "hadn't",
 'hadnt',
 "hasn't",
 'hasnt',
 "haven't",
 'havent',
 "isn't",
 'isnt',
 "mightn't",
 'mightnt',
 "mustn't",
 'mustnt',
 "needn't",
 'neednt',
 'neither',
 'never',
 'none',
 'nope',
 'nor',
 'not',
 'nothing',
 'nowhere',
 "oughtn't",
 'oughtnt',
 'rarely',
 'seldom',
 "shan't",
 'shant',
 "shouldn't",
 'shouldnt',
 'uh-uh',
 'uhuh',
 "wasn't",
 'wasnt',
 "weren't",
 'werent',
 'without',
 "won't",
 'wont',
 "wouldn't",
 'wouldnt'}