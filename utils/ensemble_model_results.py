import sys
from collections import defaultdict
from utils.calculate_results import get_query_results_dict
from utils.common_functions import make_dir

def average_similarity_over_languages(query_results_dict_per_language, languages):
    overall_query_results_dict = defaultdict(dict)

    for query in query_results_dict_per_language[0].keys():
        for query_results_dict in query_results_dict_per_language:
            for document, similarity in zip(*query_results_dict[query]):
                overall_query_results_dict[query][document] = overall_query_results_dict[query].get(document, 0) + similarity/len(languages)
        
        overall_query_results_dict[query] = sorted(overall_query_results_dict[query].items(), key=lambda x: x[1], reverse=True)
    
    
    return overall_query_results_dict
    

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        top_folder_name_suffix = args[1]
        model_languages = args[2]
        results_file_folder = args[3]
        model_type = args[4]
        layer = args[5]
        test_language = args[6]
    else:
        top_folder_name_suffix = "_train_3_9_queries_cut_after_embedding"
        model_languages = ["gujarati", "telugu", "tamil"]
        results_file_folder = "lr_0.0001_tmp_0.07_acc_1000_bs_5_3_9_3_9_epoch_best"
        model_type = "awe"
        layer = 9
        test_language = "marathi"

    model_languages_str = "_".join(model_languages)
    output_dir = f"data/{test_language}/results/{model_type}/{layer}/ensemble_{model_languages_str}{top_folder_name_suffix}/{results_file_folder}"

    print(f"Output directory: {output_dir}")
    make_dir(output_dir)

    query_results_dict_per_language = [_ for _ in range(len(model_languages))]

    for i, language in enumerate(model_languages):
        file_name = f"data/{test_language}/results/{model_type}/{layer}/{language}{top_folder_name_suffix}/{results_file_folder}/results_all.txt"
        print(f"Loading language {language} from file {file_name}")
        query_results_dict_per_language[i] = get_query_results_dict(file_name) 
    
    print(f"Calculating average similarity over languages")
    overall_query_results_dict = average_similarity_over_languages(query_results_dict_per_language, model_languages)

    print(f"Saving results to {output_dir}/results_all.txt")

    with open(f"{output_dir}/results_all.txt", "w") as f:
        for query, results in overall_query_results_dict.items():
            f.write(f"Ranking for query: {query}\n")
            f.write(f"Document: Similarity\n")
            for document, similarity in results:
                f.write(f"{document}: {similarity}\n")