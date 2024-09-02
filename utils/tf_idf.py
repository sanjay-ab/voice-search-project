"""Use TfidfVectorizer to compute the TF-IDF scores of words in a corpus. This can be
used to identify the most important words in the corpus, and hence determine
possible keywords for the corpus."""
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == "__main__":

    line_re = re.compile(r"(?P<filename>.*)\t(?P<text>.*)")

    fname = "data_misc/tamil_pruned_cleaned.txt"
    outfile = "output_words.txt"

    corpus = []
    with open(fname, "r") as f:
        for line in f:
            filename, text = re.match(line_re, line).groups()
            corpus.append(text.strip())

    max_df = 6
    sublinear_tf = True
    norm = None
    vectorizer = TfidfVectorizer(min_df=2, max_df=max_df, analyzer="word", token_pattern=r"(?u)\S\S+",
                                 use_idf=True, sublinear_tf=sublinear_tf, norm=norm)
    X = vectorizer.fit_transform(corpus)  # documents x words

    names = vectorizer.get_feature_names_out()
    vocab = vectorizer.vocabulary_

    word_tfidf = {}
    for word in names:
        # word_tfidf[word] = np.sum(X[:, vocab[word]])
        word_tfidf[word] = np.max(X[:, vocab[word]])
    word_tfidf = dict(sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True))

    with open(outfile, "w") as f:
        f.write(f"PARAMS: MAX_DF={max_df}; SUBLINEAR_TF={sublinear_tf}; NORM={bool(norm)}\n")
        for word in word_tfidf:
            f.write(f"{word}\t{word_tfidf[word]}\n")
