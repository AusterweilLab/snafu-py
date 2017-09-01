import numpy as np

animals_added=[]

with open('s2015_pairs.csv','w') as fo:
    with open('s2015.csv','r') as fi:
        fi.readline()
        for line in fi:
            line=line.split('\n')[0].split(',')
            animals=np.sort([line[1],line[2]])
            if "".join(animals) not in animals_added:
                animals_added.append("".join(animals))
                fo.write(animals[0] + "," + animals[1] + "\n")
