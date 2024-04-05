import pickle
import networkx as nx


'''struct_file = "/home/lirq/grad_design/Dataset/human/proteins.txt"
with open(struct_file, 'rb') as f:
    Z = pickle.load(f)
print(Z)'''
print("111")
graph = nx.read_gml("/home/lirq/grad_design/Dataset/human/graph.gml")
print("222", graph.order(), graph.size())