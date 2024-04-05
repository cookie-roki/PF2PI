import os
import pickle

import numpy as np

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