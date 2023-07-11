from tqdm.auto import tqdm
import csv

from string import punctuation
translator = str.maketrans('', '', punctuation)


def default_clean_text(SS, doc):
    "Input doc and return clean list of tokens"
    doc = doc.replace('\r', ' ').replace('\n', ' ')
    lower = doc.lower()  # all lower case
    nopunc = lower.translate(translator)  # remove punctuation
    words = nopunc.split()  # split into tokens
    if SS.STOPWORDS:
        words = [
            w for w in words
            if w not in SS.STOPWORDS and not w.isdigit()
        ]  # remove stopwords
    return words


def text_iterate(SS, show=False, full=False):
    'Allows iteration over a list of text or a jsonl/json file'
    if isinstance(SS.FILEINPUT, list):
        for row in tqdm(SS.FILEINPUT, total=SS.LINES,
                        disable=not show):
            yield row
    else:
        with open(SS.FILEINPUT) as f:
            reader = csv.reader(f)
            for row in tqdm(reader, total=SS.LINES, disable=not show):
                if full:
                    yield row
                else:
                    yield ' ,'.join(row)