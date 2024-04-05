import pickle

import networkx as nx

protein_file = "/home/lirq/grad_design/Dataset/human/9606.protein.info.v11.5.txt"
anno_file = "/home/lirq/grad_design/Dataset/human/goa_human.gaf"
pro_name_file = "/home/lirq/grad_design/Dataset/human/proteins.txt"
ppi_file = "/home/lirq/grad_design/Dataset/human/9606.protein.links.detailed.v11.5.txt"


def get_str_name_dict():
    f = open(protein_file, 'r')
    f.readline()
    f_data = f.readlines()
    f.close()

    protein_str_name_dict = dict()

    for line in f_data:
        elements = line.split('\t')
        protein_string_name = elements[0]
        protein_name = elements[1]
        if protein_string_name not in protein_str_name_dict:
            protein_str_name_dict[protein_string_name] = set([protein_name])
        else:
            protein_str_name_dict[protein_string_name].add(protein_name)
    return protein_str_name_dict


def get_name_uni_dict():
    f = open(anno_file, 'r')
    annot_data = f.readlines()
    f.close()

    start_line = 41
    protein_name_uni_dict = dict()

    for i in range(start_line, len(annot_data)):
        elements = annot_data[i].split('\t')
        protein_uniport = elements[1]
        protein_name = elements[2]
        if protein_name not in protein_name_uni_dict:
            protein_name_uni_dict[protein_name] = set()
            protein_name_uni_dict[protein_name].add(protein_uniport)
    return protein_name_uni_dict


def load_network(filename):
    M = []

    pro_str_name_dict = get_str_name_dict()
    pro_name_uni_dict = get_name_uni_dict()
    # print(pro_name_uni_dict)

    with open(pro_name_file, 'rb') as f:
        pro_list = pickle.load(f)

    count = 0
    graph = nx.Graph(name='combined')
    f = open(filename, 'r')
    f.readline()
    for line in f:
        print(count)
        elements = line.strip().split()
        pro1 = str(elements[0])
        pro2 = str(elements[1])
        score = float(elements[-1])

        name1 = str(pro_str_name_dict[pro1])[2:-2]
        name2 = str(pro_str_name_dict[pro2])[2:-2]

        uni1 = str(pro_name_uni_dict.get(name1, ''))[2:-2]
        uni2 = str(pro_name_uni_dict.get(name2, ''))[2:-2]

        if (not graph.has_node(pro1)) and (uni1 in pro_list):
            graph.add_node(pro1)
        if (not graph.has_node(pro2)) and (uni2 in pro_list):
            graph.add_node(pro2)
        if (score > 0) and (graph.has_node(pro1) and (graph.has_node(pro2))):
            graph.add_edge(pro1, pro2, weight=score)
        count = count + 1
    f.close()
    return graph


if __name__ == "__main__":
    graph = load_network(ppi_file)
    nx.write_gml(graph, "/home/lirq/grad_design/Dataset/human/graph.gml")