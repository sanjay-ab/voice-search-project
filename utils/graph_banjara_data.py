import matplotlib.pyplot as plt
import matplotlib.transforms
from collections import defaultdict
from utils.split_banjara_queries_documents import banjara_get_data_dicts

def banjara_graph_count_order_bar_chart(labels_dict, xlabel, ylabel, all_data_dir, queries_file, documents_file,
                                  save_fname, save_fig=False):
    labels_fnames, _ = banjara_get_data_dicts(all_data_dir, 0, 10000, queries_file, documents_file)

    labels_counts = defaultdict(int)

    for label, fnames in labels_fnames.items():
        labels_counts[label] = len(fnames)

    sorted_label_count_dict = dict(sorted(labels_counts.items(), key=lambda x: x[1], reverse=True))
    sorted_input_label_dict = {}

    for label in sorted_label_count_dict.keys():
        sorted_input_label_dict[label] = labels_dict[label]
    
    xvalues = list(sorted_input_label_dict.keys())
    yvalues = list(sorted_input_label_dict.values())

    # Create the first bar chart
    fig = plt.figure(figsize=(10, 5))
    bars = plt.bar(xvalues, yvalues)
    # bars_2 = plt.bar(single_labels_counts, single_counts, color='red')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    rot = -45
    plt.xticks(rotation=rot, va="top", ha="left", fontsize=10)

    ax = plt.gca()

    plt.tick_params(axis='x', length=10)
    plt.tick_params(axis='y', length=10)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=rot) 

    # offset transform in x direction
    dx = -3/72.; dy = 5/72. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
        # Adjust the position of x-tick labels
    ax.set_xlim(-0.4, len(labels_counts) - 0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # add numbers to top of bars
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fname, dpi=300)


if __name__ == "__main__":
    data_dir = "data/banjara/banjara_data"
    analysis_dir = "data/banjara/analysis"
    queries_file = "data/banjara/analysis/all_queries.txt"
    documents_file = "data/banjara/analysis/all_documents.txt"
    duration_min = 0
    duration_max = 10000
    save_fig = False

    labels_fnames, label_duration_dict = banjara_get_data_dicts(data_dir, duration_min, duration_max,
                                                                queries_file, documents_file)

    labels_counts = defaultdict(int)

    for label, fnames in labels_fnames.items():
        labels_counts[label] = len(fnames)

    sorted_label_duration_dict = dict(sorted(label_duration_dict.items(), key=lambda x: x[1], reverse=True))
    sorted_label_count_dict = dict(sorted(labels_counts.items(), key=lambda x: x[1], reverse=True))

    single_count_dict = {k: v for k, v in sorted_label_count_dict.items() if v == 1}

    single_labels_counts = list(single_count_dict.keys())
    single_counts = list(single_count_dict.values())

    labels_counts = list(sorted_label_count_dict.keys())
    counts = list(sorted_label_count_dict.values())

    labels_durations = list(sorted_label_duration_dict.keys())
    durations = list(sorted_label_duration_dict.values())

    # xvalues = labels_durations
    # yvalues = durations

    # xvalues = labels_counts
    # yvalues = counts

    # Create the first bar chart
    fig = plt.figure(figsize=(10, 5))
    bars = plt.bar(labels_counts, counts)
    # bars_2 = plt.bar(single_labels_counts, single_counts, color='red')
    plt.xlabel('Labels', fontsize=12)
    plt.ylabel('Number of recordings', fontsize=12)
    rot = -45
    plt.xticks(rotation=rot, va="top", ha="left", fontsize=10)

    ax = plt.gca()
    # for tick in ax.get_xticklabels():
    #     tick.set_va('bottom')
    #     tick.set_position((0, -0.1))  # Move labels slightly higher

    plt.tick_params(axis='x', length=10)
    plt.tick_params(axis='y', length=10)
    # plt.setp( ax.xaxis.get_majorticklabels(), rotation=rot, ha="left", rotation_mode="anchor")
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=rot) 

    # Create offset transform by 5 points in x direction
    dx = -3/72.; dy = 5/72. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
        # Adjust the position of x-tick labels
    ax.set_xlim(-0.4, len(labels_counts) - 0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # add numbers to top of bars
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    if save_fig:
        plt.savefig("figures/banjara_data/labels_counts.png", dpi=300)
    plt.show()
