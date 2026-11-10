"""TF-IDF feature extractor shared by the baseline and the label audit."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_tfidf() -> FeatureUnion:
    # Word n-grams capture phrases; char n-grams are robust to misspellings/teencode.
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 max_features=200_000, sublinear_tf=True)),
    ])
