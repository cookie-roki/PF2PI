import os
import numpy as np
import pandas as pd


def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr


for path, dir_list, file_list in os.walk("/home/lirq/grad_design/Dataset/node_human_processed"):
    for file_name in file_list:
        file_path = os.path.join(path, file_name)
        print(file_name)
        f = open(file_path, 'r')
        walk_data = f.readlines()
        f.close()
        walk_list = []
        for i in range(0, len(walk_data)):
            walk = walk_data[i].split()
            walk_list.append(walk)
        vectors = np.array(walk_list, dtype=float)
        average_vector = np.mean(vectors, axis=0)
        print(average_vector)
        average_vector = normalize_array(average_vector)
        data = pd.DataFrame(average_vector)
        data.to_csv("/home/lirq/grad_design/Dataset/embeddings/" + file_name + ".txt", sep=" ", index=False, header=False)
