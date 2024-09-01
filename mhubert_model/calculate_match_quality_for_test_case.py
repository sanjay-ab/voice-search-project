"""This file calculates a coarse measure of the quality of the matches for a test case.
    This measure was useful during development but is no longer used.
"""
MATCHES_FOR_Q1 = set(["s2556413_test26.pkl", "s2556413_test30.pkl"])
MATCHES_FOR_Q2 = set(["s2556413_test03.pkl", "s2556413_test06.pkl"])
MATCHES_FOR_Q3 = set(["s2556413_test18.pkl", "s2556413_test22.pkl"])
MATCHES_FOR_Q4 = set(["s2556413_test12.pkl", "s2556413_test15.pkl"])
MATCHES = (MATCHES_FOR_Q1, MATCHES_FOR_Q2, MATCHES_FOR_Q3, MATCHES_FOR_Q4)

def extract_results_from_file(file):
    q1_results = {}
    q2_results = {}
    q3_results = {}
    q4_results = {}
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            fname, similarity = line.split(": ")
            if 2<=i<=6:
                q1_results[fname] = float(similarity)
            if 14<=i<=18:
                q2_results[fname] = float(similarity)
            if 26<=i<=30:
                q3_results[fname] = float(similarity)
            if 38<=i<=42:
                q4_results[fname] = float(similarity)
    
    return q1_results, q2_results, q3_results, q4_results

def calculate_match_quality_for_single_file(file):
    results = extract_results_from_file(file)
    top_2_matches = 0
    difference_in_similarity = 0
    skip_difference_calculation = False
    for i, q_results in enumerate(results):
        results_list = list(q_results.keys())
        for x in range(2):
            if results_list[x] in MATCHES[i]: 
                top_2_matches += 1
            else:
                skip_difference_calculation = True
        if skip_difference_calculation: continue
        positive = (q_results[results_list[0]] + q_results[results_list[1]])/2
        negative = (q_results[results_list[2]] + q_results[results_list[3]] + q_results[results_list[4]])/3
        difference_in_similarity += (positive - negative)
    return top_2_matches, difference_in_similarity

if __name__ == "__main__":
    layers = list(range(13))
    pooling_method = "mean"
    window_size_ms = 1000  # in milliseconds
    stride_ms = 500  # in milliseconds
    window_stride_ms = [(1000, 500), (600, 300), (240, 80)]
    output_file = open("digit_recog_tests/results/results.txt", 'w')
    for lay in layers:
        output_file.write(f"Layer {lay}\n")
        for window_size_ms, stride_ms in window_stride_ms:
            output_file.write(f"Window size {window_size_ms} ms, stride {stride_ms} ms\n")
            results_dir = (f"digit_recog_tests/results/{lay}/{pooling_method}_pooled_win_{window_size_ms}ms_"
                                f"stride_{stride_ms}ms")
            results_file = f"{results_dir}/results.txt"
            top_2_matches, difference_in_similarity = calculate_match_quality_for_single_file(
                                                                            results_file)
            output_file.write(f"top 2 matches: {top_2_matches}, difference in similarity: {difference_in_similarity}\n")
