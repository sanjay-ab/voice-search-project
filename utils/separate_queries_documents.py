import csv

if __name__ == "__main__":
    path_prefix = "data/tamil/analysis"
    queries_fname = f"{path_prefix}/queries_times.txt"
    all_data_fname = f"{path_prefix}/tamil_pruned_cleaned.txt"
    output_queries_fname = f"{path_prefix}/all_queries.txt"
    output_document_fname = f"{path_prefix}/all_documents.txt"

    queries = set()
    with open(queries_fname, 'r') as queries_file:
        queries_reader = csv.reader(queries_file, delimiter='\t')
        for row in queries_reader:
            fname = row[0]
            queries.add(fname)

    docs = set()
    with open(all_data_fname, 'r') as all_data_file:
        all_data_reader = csv.reader(all_data_file, delimiter='\t')
        for row in all_data_reader:
            fname = row[0]
            if fname in queries:
                continue
            else:
                docs.add(fname)

    with open(output_document_fname, 'w') as f:
        for doc in docs:
            f.write(f"{doc}\n")
    
    with open(output_queries_fname, 'w') as f:
        for query in queries:
            f.write(f"{query}\n")