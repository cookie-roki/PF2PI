import os
import shutil

from data_annotation import get_proteins
from data_annotation import get_ppi_graph

protein_file = "/home/lirq/grad_design/Dataset/human/9606.protein.info.v11.5.txt"
ppi_file = "/home/lirq/grad_design/Dataset/human/9606.protein.links.detailed.v11.5.txt"
annot_path = "/home/lirq/grad_design/Dataset/human/goa_human.gaf"
source_path = "/home/lirq/grad_design/Dataset/node2vec_human"
destination_path = "/home/lirq/grad_design/Dataset/node_human_processed"

protein_name_dict = get_proteins(protein_file)
ppi_graph = get_ppi_graph(ppi_file)

protein_uniport_dict = dict()

f = open(annot_path, 'r')
annot_data = f.readlines()
f.close()
start_line = 41

for i in range(start_line, len(annot_data)):
    elements = annot_data[i].split('\t')
    protein_uniport = elements[1]
    protein_name = elements[2]
    if protein_name in protein_name_dict:
        flag = 1
        for string_name in protein_name_dict[protein_name]:
            if not ppi_graph.has_node(string_name):
                flag = 0
                break
        if flag and protein_name not in protein_uniport_dict:
            protein_uniport_dict[protein_uniport] = set()
            protein_uniport_dict[protein_uniport].add(protein_name)

for path, dir_list, file_list in os.walk(source_path):
    flag = 0
    for file_name in file_list:
        print(file_name)
        filename = file_name.split(".")[0]
        filename = str(filename)
        if filename in protein_uniport_dict:
            print(protein_uniport_dict[filename])
            source_file = os.path.join(source_path, str(file_name))
            shutil.move(source_file, destination_path)
