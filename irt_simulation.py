import networkx as nx
from rw.rw import *
import csv

num_graphs=100
num_walks=100
num_nodes=50

ws_link_neighbors=6
ws_probability_of_rewiring=.2
ws_edges=(ws_link_neighbors/2.0)*num_nodes

er_max_edges=((num_nodes**2)-num_nodes)/2.0
er_probability_of_edge_creation=ws_edges/float(er_max_edges) # mean edges == edges in ws graph

er_first_hit_list=[]
ws_first_hit_list=[]

for i in range(num_graphs):
        print "graph", i
        # generate connected ws and er graphs
        ws_graph=nx.connected_watts_strogatz_graph(num_nodes, ws_link_neighbors, ws_probability_of_rewiring, 10000)
        while True:
            er_graph=nx.erdos_renyi_graph(num_nodes, er_probability_of_edge_creation)
            if list(nx.connected_component_subgraphs(er_graph))[0].number_of_nodes() == num_nodes:
                break
        
        for j in range(num_walks):
            er_walk=random_walk(er_graph)
            ws_walk=random_walk(ws_graph)
            er_first_hit=list(zip(*firstHit(er_walk))[1])
            ws_first_hit=list(zip(*firstHit(ws_walk))[1])

            er_first_hit_list.append(er_first_hit)
            ws_first_hit_list.append(ws_first_hit)

with open('irt_simulation_data.csv','w') as csvfile:
    f=csv.writer(csvfile,delimiter=',')
    f.writerow(['type'] + ['i'+str(i) for i in range(1,num_nodes+1)])
    for i in er_first_hit_list:
        f.writerow(['er'] + i)
    for i in ws_first_hit_list:
        f.writerow(['ws'] + i)
