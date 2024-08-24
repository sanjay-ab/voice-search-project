import os
from mhubert_model.query_document_search import Ranker
from utils.common_functions_pytorch import print_memory_usage

def collate_embeddings(embedding_dir, layer, window_size_ms, stride_ms,
                       pooling_method="mean"):
    document_prefix = f"{embedding_dir}/documents"

    if pooling_method == "none" or pooling_method is None:
        document_embedded_states_dir = (f"{document_prefix}/{layer}/raw")
    else:
        document_embedded_states_dir = (f"{document_prefix}/{layer}/{pooling_method}_pooled_win_"
                f"{window_size_ms}ms_stride_{stride_ms}ms")

    if not os.path.exists(document_embedded_states_dir):
            raise ValueError((f"Document embedded states directory"
                            f"{document_embedded_states_dir} does not exist"))
    
    ranker = Ranker(document_embedded_states_dir, force_dont_use_all_embeddings_file=True)
    print_memory_usage()
    ranker.save_document_embedding()

if __name__ == "__main__":
    embedding_dir = "data/tamil/embeddings"
    layer = "9"
#     layer = sys.argv[1]
    pooling_method = "mean"
    window_size_ms = 240  # in milliseconds
    stride_ms = 80  # in milliseconds
    collate_embeddings(embedding_dir, layer, window_size_ms, stride_ms, 
                       pooling_method=pooling_method)