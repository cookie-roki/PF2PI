"""
Created on Thu Jan 18 2024

Node2vec algorithm

@author: Ruiqi Li
"""
import os
import random
import networkx as nx
import pandas as pd


def load_con_graph(filename):
    graph = nx.Graph(name='contact')
    f = open(filename, 'r')
    for line in f:
        nodes = line.strip().split()
        node1 = nodes[0]
        node2 = nodes[1]
        if not graph.has_node(node1):
            graph.add_node(node1)
        if not graph.has_node(node2):
            graph.add_node(node2)
        graph.add_edge(node1, node2, weight=1)
    f.close()
    return graph


def node2vec_walk(graph, node, sequence_length, p, q):
    walk = [str(node)]
    for _ in range(sequence_length - 1):
        neighbors = list(graph.neighbors(node))
        if len(neighbors) > 0:
            probabilities = [1 / p if neighbor == walk[-1] else 1 if neighbor in neighbors else 1 / q for neighbor in
                             neighbors]
            selected_neighbor = random.choices(neighbors, weights=probabilities)[0]
            walk.append(str(selected_neighbor))
            node = selected_neighbor
        else:
            break
    return walk


def generate_node2vec_sequence(graph, sequence_length, p, q):
    sequence = []
    for node in graph.nodes():
        walk = node2vec_walk(graph, node, sequence_length, p, q)
        sequence.append(walk)
    return sequence


for path, dir_list, file_list in os.walk("/home/lirq/grad_design/Dataset/edges_human"):
    for file_name in file_list:
        print(file_name)
        graph = load_con_graph(os.path.join(path, file_name))
        result = generate_node2vec_sequence(graph, sequence_length=30, p=0.8, q=1.2)
        data = pd.DataFrame(result)
        filename = file_name.split(".")
        name = filename[0]
        data.to_csv("/home/lirq/grad_design/Dataset/node2vec_human/" +name + ".txt", sep=" ", index=False, header=False)