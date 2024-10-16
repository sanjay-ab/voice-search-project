"""Use to symlink files from source directories to target directories, based on a list of filenames."""
import os

def make_dir(path):
    if not os.path.exists(f"{path}"):
        os.makedirs(f"{path}")

def link_files(fname_with_list, source_dirs, target_dir, fname_extension="wav"):
    """Link files specified in file "fname_with_list" from dir in source_dirs to target_dir.
    File fname_with_list should contain a list of filename basenames, one per line.

    Args:
        fname_with_list (str): path of file with a list of filenames to link
        source_dirs (list[str]): list of source directories that contain the files to link
        target_dir (str): target directory to link files to
    """
    directory_files_dict = {}
    for directory in source_dirs:
        directory_files_dict[directory] = set(os.listdir(directory))

    with open(fname_with_list, 'r') as f:
        for row in f:
            fname_basename = row.strip()
            for directory in source_dirs:
                if f"{fname_basename}.{fname_extension}" in directory_files_dict[directory]:
                    os.symlink(f"../../../{directory}/{fname_basename}.{fname_extension}", 
                        f"{target_dir}/{fname_basename}.{fname_extension}")
                    break
            else:
                print(f"File {fname_basename}.{fname_extension} not found in source directories: {source_dirs}")

if __name__ == "__main__":
    language = "gujarati"
    top_level_dir = f"data/{language}/"
    test_data_dir = f"{top_level_dir}/all_data"
    # test_data_dir = f"{top_level_dir}/test/audio"
    # train_data_dir = f"{top_level_dir}/train/audio"
    train_data_dir = test_data_dir
    queries_fname = f"{top_level_dir}/analysis/all_queries.txt"
    documents_fname = f"{top_level_dir}/analysis/all_documents.txt"
    training_data_fname = f"{top_level_dir}/analysis/training_data.txt"
    validation_data_fname = f"{top_level_dir}/analysis/validation_data.txt"
    output_query_dir = f"{top_level_dir}/queries"
    output_document_dir = f"{top_level_dir}/documents"
    output_training_dir = f"{top_level_dir}/training_data"
    output_validation_dir = f"{top_level_dir}/validation_data"

    # only link queries for banjara data - for tamil, queries are separately cut out of documents
    link_queries = False  
    link_documents = False
    link_training_data = False
    link_validation_data = False

    if link_queries:
        make_dir(output_query_dir)
        link_files(queries_fname, [test_data_dir, train_data_dir], output_query_dir)    

    if link_documents:
        make_dir(output_document_dir)
        link_files(documents_fname, [test_data_dir, train_data_dir], output_document_dir)

    if link_training_data:
        make_dir(output_training_dir)
        link_files(training_data_fname, [test_data_dir, train_data_dir], output_training_dir)
    
    if link_validation_data:
        link_files(validation_data_fname, [test_data_dir, train_data_dir], output_validation_dir)
        # embedding_dir = f"{top_level_dir}/embeddings/validation_data/9/raw_all"
        # output_validation_dir = f"{top_level_dir}/embeddings/validation_data/9/raw"
        # make_dir(output_validation_dir)
        # # nk_files(validation_data_fname, [embedding_dir], output_validation_dir, "pkl")