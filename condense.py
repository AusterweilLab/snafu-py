# temporary script to take full USF network and generate USF' network
# USF' network includes edges between nodes that are indirectly connected in USF,
# even if nodes are not directly connected

import sys
sys.path.append('./rw')
import rw

usf_full, allitems = rw.read_csv('USF_full.csv')
usf_animal, animals = rw.read_csv('USF_animal_subset.csv')
