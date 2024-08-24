from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

if __name__ == "__main__":

    fname = r"../data/LibriSpeech-84-121123.trans.txt"
    corpus = []
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            line = line[15:]
            corpus.append(line)
    
    vectorizer = TfidfVectorizer(min_df=2, max_df=6, sublinear_tf=True)
    X = vectorizer.fit_transform(corpus)  # documents x words
    print(X)
    names = vectorizer.get_feature_names_out()
    vocab = vectorizer.vocabulary_
    word_tfidf = {}
    for word in names:
        word_tfidf[word] = np.mean(X[:, vocab[word]])
    word_tfidf = dict(sorted(word_tfidf.items(), key=lambda x: x[1]))
    for word in word_tfidf:
        print(word, word_tfidf[word])
    
    # Iterate over the documents
    docs = []
    for i in range(len(corpus)):
        # If the word's TF-IDF score is greater than zero, add the document to our list
        if X[i, vocab["morrel"]] > 0:
            docs.append(corpus[i])
    print(docs)
