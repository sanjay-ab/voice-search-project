"""Use to create a list of all documents in a language search collection."""
import os

if __name__ == "__main__":

    language = "odia"
    folder = "documents"
    out_file = f"data/{language}/analysis/all_{folder}.txt"

    if folder == "documents" and language in ["gujarati", "hindi", "marathi", "telugu", "odia"]:
        corpus_file = f"data/{language}/analysis/search_corpus.txt"
        files = set()
        with open(corpus_file, "r") as f:
            for line in f:
                line = line.split("\t")
                files.add(line[0].strip())
        with open(out_file, "w") as f:
            for file in sorted(files):
                f.write(f"{file}\n")
