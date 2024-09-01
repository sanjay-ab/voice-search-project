"""If results are saved over multiple files, this script collates them into a single file."""
import os

from mhubert_model.query_document_search import get_embedding_and_results_dir

if __name__ == "__main__":
    results_dir_prefix = f"tamil_results"
    layer = "9"
    pooling_method = "mean"
    window_size_ms = 120  # in milliseconds
    stride_ms = 40  # in milliseconds
    n_parts = 64
    query_multiple_vectors = True
    all_results_fname = "results_all.txt"
    remove_parts = True

    _, _, results_dir = get_embedding_and_results_dir("", "", results_dir_prefix, pooling_method, 
                            layer, window_size_ms, stride_ms, query_multiple_vectors)

    overall_results_file = f"{results_dir}/{all_results_fname}"

    with open(overall_results_file, "w") as f:
        for i in range(n_parts):
            part_results_file = f"{results_dir}/results_{i}.txt"
            with open(part_results_file, "r") as part_f:
                for line in part_f:
                    f.write(line)
    
    if remove_parts:
        files = os.listdir(results_dir)
        files.remove(all_results_fname)
        for file in files:
            os.remove(f"{results_dir}/{file}")