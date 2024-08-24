import argparse
from multiprocessing import Pool
import sys
import numpy as np
from tqdm import tqdm
import os

def knn_pairs_to_cluster_pairs(arg):

    """
    Translate KNN output to cluster composed only of pairs.
    """
    print(arg)
    pairs, mapping_file, output_path,randint =arg
    verbose=False
    # Create mapping
    mapping = dict()
    with open(mapping_file, 'r') as f:
        for l in f:
            id, name = l.strip().split()
            mapping[int(id)] = name

    # Iterate through pairs
    current_class = 0
    with open(pairs, 'r') as f:
        with open(output_path, 'w') as output:

            lines = f.readlines()
            n_lines = len(lines)
            print("total number of lines : %s" % n_lines)

            for l in tqdm(lines, disable=(not verbose)):
                elements = l.strip().split()
                if len(elements) != 7 :
                    continue

                f1, f2, s1, e1, s2, e2, d = elements
                f1, f2, s1, e1, s2, e2 = map(int, [f1, f2, s1, e1, s2, e2])
                #if float(d)<0.90:
                #    continue
                if randint>1:
                    if np.random.randint(0,100)>randint:
                        continue

                output.write('{filename} {begin} {end} '.format(
                    filename=mapping[f1],
                    begin=s1 / 100.0, end=e1 / 100.0))
                output.write('{filename} {begin} {end} {d}'.format(
                    filename=mapping[f2],
                    begin=s2 / 100.0, end=e2 / 100.0, d=d))
                output.write("\n")

                current_class += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pairs_folder', metavar='pairs_folder',
            help='Path to folder containing pairs')
    parser.add_argument('mapping_file', metavar='mapping_file',
            help='Path to file containing file name mapping')
    parser.add_argument('clusters_folder', metavar='clusters_folder',
            help='Path to output file')
    parser.add_argument('n_cpus', type=int,
            help='nb on cpus')
    parser.add_argument('nb_pairs', type=int,
            help='nb of pairs in total')
    

    args = parser.parse_args()
    pairs_folder = args.pairs_folder
    clusters_folder = args.clusters_folder
    mapping_file = args.mapping_file

    max_pairs=50000000
    randint=int(100*max_pairs/args.nb_pairs)
    
    arg=[(os.path.join(pairs_folder,f),mapping_file,os.path.join(clusters_folder,f),randint) for f in os.listdir(pairs_folder)]    
    
    #for a in arg:
    #    knn_pairs_to_cluster_pairs(a)
    pool = Pool(processes = args.n_cpus)
    pool.map(knn_pairs_to_cluster_pairs,arg)
    pool.close()


