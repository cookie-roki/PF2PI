"""
Created on Fri Jun 5 2024

annotation_preprocess
part of code borrowed from CFAGO

@author: Ruiqi Li
"""
import argparse
import os.path
import numpy as np
import scipy.io as sio
import networkx as nx

from get_ppi_graph import load_network


def get_proteins(protein_file):
    f = open(protein_file, 'r')
    f.readline()
    f_data = f.readlines()
    f.close()

    protein_name_dict = dict()

    for line in f_data:
        elements = line.split('\t')
        protein_string_name = elements[0]
        protein_name = elements[1]
        if protein_name not in protein_name_dict:
            protein_name_dict[protein_name] = set([protein_string_name])
        else:
            protein_name_dict[protein_name].add(protein_string_name)
    return protein_name_dict


'''def get_ppi_graph(ppi_file):
    f = open(ppi_file, 'r')
    f.readline()
    f_data = f.readlines()
    f.close()
    ppi_graph = nx.Graph(name='ppi')

    for line in f_data:
        elements = line.split()
        protein_string_name = elements[0]
        if not ppi_graph.has_node(protein_string_name):
            ppi_graph.add_node(protein_string_name)
        protein_string_name = elements[1]
        if not ppi_graph.has_node(protein_string_name):
            ppi_graph.add_node(protein_string_name)

    print("the number of ppi node is: ", ppi_graph.number_of_nodes())
    return ppi_graph'''


def get_go_annots(protein_name_dict, ppi_graph, aspect, evidences, annot_file, train_date, valid_date, start_line):
    f = open(annot_file, 'r')
    annot_data = f.readlines()
    f.close()

    train_go_annot_dict = dict()
    valid_go_annot_dict = dict()
    test_go_annot_dict = dict()

    for i in range(start_line, len(annot_data)):
        elements = annot_data[i].split('\t')
        protein_name = elements[2]
        go_id = elements[4]
        evid = elements[6]
        go_aspect = elements[8]
        date = elements[13]
        if protein_name in protein_name_dict:
            flag = 1
            for string_name in protein_name_dict[protein_name]:
                if not ppi_graph.has_node(string_name):
                    flag = 0
                    break
            if go_aspect == aspect and evid in evidences and flag:
                if int(date) <= int(train_date):
                    if go_id not in train_go_annot_dict:
                        train_go_annot_dict[go_id] = set()
                    train_go_annot_dict[go_id].add(protein_name)
                elif int(date) <= int(valid_date):
                    if go_id not in valid_go_annot_dict:
                        valid_go_annot_dict[go_id] = set()
                    valid_go_annot_dict[go_id].add(protein_name)
                else:
                    if go_id not in test_go_annot_dict:
                        test_go_annot_dict[go_id] = set()
                    test_go_annot_dict[go_id].add(protein_name)
    annots = {
        'train': train_go_annot_dict,
        'valid': valid_go_annot_dict,
        'test': test_go_annot_dict
    }
    return annots


def remove_someTerms(annots):
    train_dict = annots['train']
    valid_dict = annots['valid']
    test_dict = annots['test']

    for go_id in list(train_dict):
        if go_id not in valid_dict:
            train_dict.pop(go_id)

    for go_id in list(train_dict):
        if go_id not in test_dict:
            train_dict.pop(go_id)

    for go_id in list(valid_dict):
        if go_id not in train_dict:
            valid_dict.pop(go_id)

    for go_id in list(valid_dict):
        if go_id not in test_dict:
            valid_dict.pop(go_id)

    for go_id in list(test_dict):
        if go_id not in train_dict:
            test_dict.pop(go_id)

    for go_id in list(test_dict):
        if go_id not in valid_dict:
            test_dict.pop(go_id)

    annos = {
        'train': train_dict,
        'valid': valid_dict,
        'test': test_dict
    }
    return annos


def get_proteinsSet(annot_dict, go_terms):
    protein_set = set()
    for key in go_terms:
        protein_set = protein_set.union(annot_dict[key])
    return protein_set


def annotations_process(protein_name_dict, ppi_graph, evidences, annot_file, train_date, valid_date, data_path, org,
                        start_line):
    Annots = {
        'GO': {
            'P': {'train': [], 'valid': [], 'test': []},
            'F': {'train': [], 'valid': [], 'test': []},
            'C': {'train': [], 'valid': [], 'test': []}
        },
        'index': {
            'P': {'train': [], 'valid': [], 'test': []},
            'F': {'train': [], 'valid': [], 'test': []},
            'C': {'train': [], 'valid': [], 'test': []}
        },
        'labels': {
            'P': {'terms': []},
            'F': {'terms': []},
            'C': {'terms': []}
        }
    }

    ppi_nodes = list(ppi_graph.nodes())
    proteins_num = len(ppi_nodes)

    for aspect in ['P', 'F', 'C']:
        annots = get_go_annots(protein_name_dict, ppi_graph, aspect, evidences, annot_file, train_date, valid_date,
                               start_line)
        annos = remove_someTerms(annots)
        print("number of proteins is:", proteins_num)

        train_annot_dict = annos['train']
        valid_annot_dict = annos['valid']
        test_annot_dict = annos['test']

        go_terms = []
        for go_id in list(train_annot_dict):
            go_term_proteins_num = len(train_annot_dict[go_id])
            protein_ratio = go_term_proteins_num / proteins_num
            if go_term_proteins_num >= 10 and protein_ratio <= 0.05:
                go_terms.append(go_id)
        print("number of GO before preprocess is:", aspect, ": ", len(go_terms))

        train_proteins = get_proteinsSet(train_annot_dict, go_terms)
        pre_valid_annot_dict = valid_annot_dict
        for go_id in valid_annot_dict:
            pre_valid_annot_dict[go_id] = valid_annot_dict[go_id].difference(train_proteins)

        for go_id in list(valid_annot_dict):
            go_term_proteins_num = len(pre_valid_annot_dict[go_id])
            protein_ratio = go_term_proteins_num / proteins_num
            if go_term_proteins_num < 5 or protein_ratio > 0.05:
                if go_id in go_terms:
                    go_terms.remove(go_id)

        train_proteins = get_proteinsSet(train_annot_dict, go_terms)
        pre_valid_annot_dict = valid_annot_dict
        for go_id in valid_annot_dict:
            pre_valid_annot_dict[go_id] = valid_annot_dict[go_id].difference(train_proteins)
        valid_proteins = get_proteinsSet(pre_valid_annot_dict, go_terms)

        no_test_proteins = train_proteins.union(valid_proteins)
        pre_test_annot_dict = test_annot_dict
        for go_id in test_annot_dict:
            pre_test_annot_dict[go_id] = test_annot_dict[go_id].difference(no_test_proteins)

        for go_id in list(test_annot_dict):
            go_term_proteins_num = len(pre_test_annot_dict[go_id])
            protein_ratio = go_term_proteins_num / proteins_num
            if go_term_proteins_num < 1 or protein_ratio > 0.05:
                if go_id in go_terms:
                    go_terms.remove(go_id)

        train_proteins = get_proteinsSet(train_annot_dict, go_terms)
        pre_valid_annot_dict = valid_annot_dict
        for go_id in valid_annot_dict:
            pre_valid_annot_dict[go_id] = valid_annot_dict[go_id].difference(train_proteins)
        valid_proteins = get_proteinsSet(pre_valid_annot_dict, go_terms)

        no_test_proteins = train_proteins.union(valid_proteins)
        pre_test_annot_dict = test_annot_dict
        for go_id in test_annot_dict:
            pre_test_annot_dict[go_id] = test_annot_dict[go_id].difference(no_test_proteins)
        test_proteins = get_proteinsSet(test_annot_dict, go_terms)

        go_term_num = len(go_terms)
        print("number of go terms is: ", aspect, ": ", go_term_num)

        print("number of train proteins after preprocess is: ", aspect, ": ", len(train_proteins))
        print("number of validation proteins after preprocess is: ", aspect, ": ", len(valid_proteins))
        print("number of test proteins after preprocess is: ", aspect, ": ", len(test_proteins))

        Annots['labels'][aspect]['terms'] = go_terms
        go_index_dict = dict(zip(go_terms, range(go_term_num)))

        aa_train_set = set()
        aa_valid_set = set()
        aa_test_set = set()
        for go_id in go_terms:
            for protein_name in train_annot_dict[go_id]:
                aa_train_set.add(protein_name)
                protein_string_name = protein_name_dict[protein_name]
                for str_name in protein_string_name:
                    protein_id = ppi_nodes.index(str_name)
                    go_index = go_index_dict[go_id]
                    if protein_id not in Annots['index'][aspect]['train']:
                        Annots['index'][aspect]['train'].append(protein_id)
                        Annots['GO'][aspect]['train'].append(np.zeros(go_term_num, dtype=np.int64).tolist())
                    protein_index = Annots['index'][aspect]['train'].index(protein_id)
                    Annots['GO'][aspect]['train'][protein_index][go_index] = 1
            for protein_name in pre_valid_annot_dict[go_id]:
                aa_valid_set.add(protein_name)
                protein_string_name = protein_name_dict[protein_name]
                for str_name in protein_string_name:
                    protein_id = ppi_nodes.index(str_name)
                    go_index = go_index_dict[go_id]
                    if protein_id not in Annots['index'][aspect]['valid']:
                        Annots['index'][aspect]['valid'].append(protein_id)
                        Annots['GO'][aspect]['valid'].append(np.zeros(go_term_num, dtype=np.int64).tolist())
                    protein_index = Annots['index'][aspect]['valid'].index(protein_id)
                    Annots['GO'][aspect]['valid'][protein_index][go_index] = 1
            for protein_name in pre_test_annot_dict[go_id]:
                aa_test_set.add(protein_name)
                protein_string_name = protein_name_dict[protein_name]
                for str_name in protein_string_name:
                    protein_id = ppi_nodes.index(str_name)
                    go_index = go_index_dict[go_id]
                    if protein_id not in Annots['index'][aspect]['test']:
                        Annots['index'][aspect]['test'].append(protein_id)
                        Annots['GO'][aspect]['test'].append(np.zeros(go_term_num, dtype=np.int64).tolist())
                    protein_index = Annots['index'][aspect]['test'].index(protein_id)
                    Annots['GO'][aspect]['test'][protein_index][go_index] = 1
        print("---number of train proteins after preprocess is: ", aspect, ": ", len(aa_train_set))
        print("---number of validation proteins after preprocess is: ", aspect, ": ", len(aa_valid_set))
        print("---number of test proteins after preprocess is: ", aspect, ": ", len(aa_test_set))

    save_file = os.path.join(data_path, org, org + '_annot_2.mat')
    sio.savemat(save_file, Annots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data_path', type=str, help="the data path")
    parser.add_argument('-af', '--annotation_file', type=str, help="the annotation file")
    parser.add_argument('-pf', '--protein_file', type=str, help="the protein file")
    parser.add_argument('-ppif', '--ppi_file', type=str, help="the ppi file")
    parser.add_argument('-org', '--organism', type=str, help="the organism")
    parser.add_argument('-stl', '--start_line', type=int, help="the start line of the annotation file")

    args = parser.parse_args()
    annotation_file = os.path.join(args.data_path, args.organism, args.annotation_file)
    protein_file = os.path.join(args.data_path, args.organism, args.protein_file)
    ppi_file = os.path.join(args.data_path, args.organism, args.ppi_file)
    path = args.data_path

    if not os.path.exists(path):
        os.mkdir(path)

    evidences = {'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC'}
    train_date = '20171231'
    valid_date = '20201231'

    protein_name_dict = get_proteins(protein_file)
    ppi_graph = load_network(ppi_file)
    annotations_process(protein_name_dict, ppi_graph, evidences, annotation_file, train_date, valid_date,
                        args.data_path, args.organism, args.start_line)
