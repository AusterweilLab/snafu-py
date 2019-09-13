# need to install snafu! 
# pip install git+https://github.com/AusterweilLab/snafu-py
import snafu
import numpy as np
import random
import networkx as nx


# Load animal component of USF semantic network (Nelson et al, 1999)
usf_network, usf_items = snafu.read_graph('snet/usf_animal_subset.snet')

# Perturb this network by randomly changing ~10% of the edges to non-edges and an equivalent
# number of non-edges to edges
edges = list(zip(*np.where(usf_network==1.0)))      # edges in USF network
nonedges = list(zip(*np.where(usf_network==0.0)))  # non-edges in USF network
nonedges = [i for i in nonedges if i[0] != i[1]]  # exclude diagonal

# random sample of edges / non-edges
n = round(len(edges)*.1)
edges_to_flip = random.sample(edges, n)
nonedges_to_flip = random.sample(nonedges, n)

# copy USF network and flip edges
alternate_network = np.copy(usf_network)
alternate_network[list(zip(*edges_to_flip))] = 0.0     # flip edges to nonedges
alternate_network[list(zip(*nonedges_to_flip))] = 1.0  # flip nonedges to nonedges


# generate fake fluency data from the USF network
datamodel = snafu.DataModel({
                'jump':         0.05,           # allow 5% of jumping on each step
                'jump_type':   'stationary',    # when jumping, jump to a random node proportional to node degree
                'numx':         20,             # generate 10 fluency lists
                'trim':         35})            # each fluency list should be 20 items long 

# 10 lists from USF network
usf_lists = snafu.gen_lists(nx.from_numpy_array(usf_network), datamodel)[0]
alternate_lists = snafu.gen_lists(nx.from_numpy_array(alternate_network), datamodel)[0]

# Calculate probability of each list from each network
p_usf_from_usf = snafu.probX(usf_lists, usf_network, datamodel)[0]
p_alternate_from_usf = snafu.probX(alternate_lists, usf_network, datamodel)[0]
p_usf_from_alternate = snafu.probX(usf_lists, alternate_network, datamodel)[0]
p_alternate_from_alternate = snafu.probX(alternate_lists, alternate_network, datamodel)[0]

print ('Log-likelihood of generating USF lists from USF network: ', p_usf_from_usf)
print ('Log-likelihood of generating USF lists from alternate network: ', p_usf_from_alternate)
print('')
print ('Log-likelihood of generating alternate lists from USF network: ', p_alternate_from_usf)
print ('Log-likelihood of generating alternate lists from alternate network: ', p_alternate_from_alternate)
