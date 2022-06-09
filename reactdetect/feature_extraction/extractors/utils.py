import re
import string


# this dictionary will hold key value pairs of:
# (tag, list of feature extraction function objects with this tag)
EXTR_FCNS_BY_TAG = {}

# this dict will hold key value pairs of (function name, function object)
EXTR_FCNS_BY_NAME = {}


def apply_tags(tags):
    """Decorator function that tags a function with the supplied tags
    and then creates a wrapper around the function and returns this
    wrapper function."""

    # make sure tags is a list
    assert isinstance(tags, list), f'tags is a {type(tags)} and it needs to be a list'

    def f_wrapper(f):
        # register function by name
        EXTR_FCNS_BY_NAME[f.__name__] = f

        # register function by its tags
        for tag in tags:
            if tag in EXTR_FCNS_BY_TAG:
                EXTR_FCNS_BY_TAG[tag].append(f)
            else:
                EXTR_FCNS_BY_TAG[tag] = [f]

        return f

    return f_wrapper


def swap(w, i, j):
    """
    Swap two chars. at indices i and j for word w.
    """
    w = list(w)
    w[i], w[j] = w[j], w[i]
    return ''.join(w)


def delete(w, i):
    """
    Delete a char. at index i for word w.
    """
    w = list(w)
    del w[i]
    return ''.join(w)


def insert(w, i, c):
    """
    Insert char. c at index i for word w.
    """
    return w[:i] + c + w[i:]


def replace(w, i, c):
    """
    Replace char. at index i with char. c for word w.
    """
    w = list(w)
    w[i] = c
    return ''.join(w)


def remove_punctuation(s):
    """
    Remove all punctuation from the string s.
    """
    return s.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(s):
    """
    Remove all numbers from the string s.
    """
    return ''.join(c for c in s if not c.isdigit())


def split_sentences(st):
    """
    Split a string into its component sentences
    """
    pieces = re.split(r'([.?!])\s*', st)
    sentences = []
    sentence = ''
    for i, piece in enumerate(pieces):
        sentence += piece
        if len(piece) <= 1:
            sentences.append(sentence)
            sentence = ''
    if sentence != '':
        sentences.append(sentence)
    sentences_ = []
    for sentence in sentences:
        if sentence != '':
            sentences_.append(sentence)
    sentences = sentences_
    if len(sentences) == 0 or sentences[-1]:
        return sentences
    else:
        return sentences[:-1]
