"""
Feature engineering for the Quora duplicate-question model.

This mirrors the notebook (d1.ipynb) step by step so that training and
inference use exactly the same features:

  22 engineered features  +  3000 BoW (question1)  +  3000 BoW (question2)

Notes on fidelity with the notebook:
  * Engineered features are computed on the RAW question text.
  * Bag-of-words is computed on punctuation-stripped text.
  * Stop-word matching is case-sensitive (as in the notebook).
  * NLTK's English stop-word list is embedded so no runtime download is needed.
  * `distance.lcsubstrings` is replaced with difflib (same longest-substring length).
"""
import string
from difflib import SequenceMatcher

import numpy as np
from fuzzywuzzy import fuzz

NLTK_STOPWORDS = frozenset(['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"])

SAFE_DIV = 0.0001

ENGINEERED_FEATURE_NAMES = [
    "q1_len", "q2_len", "q1_word_count", "q2_word_count",
    "word_total", "common_words", "word_share",
    "cwc_min", "cwc_max", "csc_min", "csc_max", "ctc_min", "ctc_max",
    "last_word_eq", "first_words_eq",
    "mean_length", "abs_length_diff", "longest_substr",
    "fuzz_ratio", "fuzz_partial_ratio", "tokensort_ratio", "tokenset_ratio",
]

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def remove_punc(text) -> str:
    if isinstance(text, str):
        return text.translate(_PUNCT_TABLE)
    return ""


def _word_sets(q1: str, q2: str):
    w1 = {w.lower().strip() for w in q1.split(" ")}
    w2 = {w.lower().strip() for w in q2.split(" ")}
    return w1, w2


def basic_features(q1: str, q2: str):
    w1, w2 = _word_sets(q1, q2)
    total = len(w1) + len(w2)
    common = len(w1 & w2)
    share = round(common / total, 2) if total else 0
    return [
        len(q1), len(q2),
        len(q1.split(" ")), len(q2.split(" ")),
        total, common, share,
    ]


def token_features(q1: str, q2: str):
    f = [0.0] * 8
    t1, t2 = q1.split(), q2.split()
    if len(t1) == 0 or len(t2) == 0:
        return f

    s1 = {w for w in t1 if w not in NLTK_STOPWORDS}
    s2 = {w for w in t2 if w not in NLTK_STOPWORDS}
    st1 = {w for w in t1 if w in NLTK_STOPWORDS}
    st2 = {w for w in t2 if w in NLTK_STOPWORDS}

    common_stop = len(st1 & st2)
    common_word = len(s1 & s2)
    common_token = len(set(t1) & set(t2))

    f[0] = common_word / (min(len(s1), len(s2)) + SAFE_DIV)
    f[1] = common_word / (max(len(s1), len(s2)) + SAFE_DIV)
    f[2] = common_stop / (min(len(st1), len(st2)) + SAFE_DIV)
    f[3] = common_stop / (max(len(st1), len(st2)) + SAFE_DIV)
    f[4] = common_token / (min(len(t1), len(t2)) + SAFE_DIV)
    f[5] = common_token / (max(len(t1), len(t2)) + SAFE_DIV)
    f[6] = int(t1[-1] == t2[-1])
    f[7] = int(t1[0] == t2[0])
    return f


def length_features(q1: str, q2: str):
    f = [0.0] * 3
    t1, t2 = q1.split(), q2.split()
    if len(t1) == 0 or len(t2) == 0:
        return f
    f[0] = (len(t1) + len(t2)) / 2
    f[1] = abs(len(t1) - len(t2))
    # length of the longest common substring (same value distance.lcsubstrings gives)
    m = SequenceMatcher(None, q1, q2, autojunk=False).find_longest_match(0, len(q1), 0, len(q2))
    f[2] = m.size / (min(len(q1), len(q2)) + 1) if m.size > 0 else 0
    return f


def fuzzy_features(q1: str, q2: str):
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2),
    ]


def engineered_features(q1: str, q2: str):
    """The 22 hand-crafted features, in the notebook's column order."""
    return (
        basic_features(q1, q2)
        + token_features(q1, q2)
        + length_features(q1, q2)
        + fuzzy_features(q1, q2)
    )


def build_feature_row(q1: str, q2: str, cv) -> np.ndarray:
    """Full feature vector (1, 22 + 2*len(vocab)) for one question pair."""
    eng = np.array(engineered_features(q1, q2), dtype=np.float32).reshape(1, -1)
    b1 = cv.transform([remove_punc(q1)]).toarray().astype(np.float32)
    b2 = cv.transform([remove_punc(q2)]).toarray().astype(np.float32)
    return np.hstack((eng, b1, b2))
