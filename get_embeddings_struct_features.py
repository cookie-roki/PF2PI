import os
import numpy as np
import pandas as pd
import pickle

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

for path, dir_list, file_list in os.walk("/home/lirq/grad_design/Dataset/embeddings"):
    embeddings_list = []
    pro_name_list = []
    for file_name in file_list:
        file_path = os.path.join(path, file_name)
        file_name = file_name.split(".")[0]
        print(file_name)
        pro_name_list.append(file_name)
        f = open(file_path, 'r')
        embedding = f.readlines()
        f.close()
        embedding_list = []
        for i in range(0, len(embedding)):
            embedding_list.append(float(embedding[i].strip()))
        embeddings_list.append(embedding_list)
    embeddings_array = np.array(embeddings_list)
    print(type(embeddings_array), embeddings_array.shape)
    s_files = "/home/lirq/grad_design/Dataset/human/struct_features.npy"
    with open(s_files, 'wb') as ff:
        pickle.dump(embeddings_array, ff)
    s_file = "/home/lirq/grad_design/Dataset/human/proteins.txt"
    with open(s_file, 'wb') as f:
        pickle.dump(pro_name_list, f)