from core import *

import numpy as np

try: import matplotlib.pyplot as plt
except: print "Warning: Failed to import matplotlib"

# write graph to GraphViz file (.dot)
def drawDot(g, filename, labels={}):
    if type(g) == np.ndarray:
        g=nx.to_networkx_graph(g)
    if labels != {}:
        nx.relabel_nodes(g, labels, copy=False)
    nx.drawing.write_dot(g, filename)

# draw graph
def drawG(g,Xs=[],labels={},save=False,display=True):
    if type(g) == np.ndarray:
        g=nx.to_networkx_graph(g)
    nx.relabel_nodes(g, labels, copy=False)
    #pos=nx.spring_layout(g, scale=5.0)
    pos = nx.graphviz_layout(g, prog="fdp")
    nx.draw_networkx(g,pos,node_size=1000)
#    for node in range(numnodes):                    # if the above doesn't work
#        plt.annotate(str(node), xy=pos[node])       # here's a workaround
    if Xs != []:
        plt.title(Xs)
    plt.axis('off')
    if save==True:
        plt.savefig('temp.png')                      # need to parameterize
    if display==True:
        plt.show()

def drawMat(mat,mat2=0,cmap=plt.cm.ocean):
    if mat2:
        mat=np.array(mat2)-np.array(mat)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(mat, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.ion()
    plt.show()
    return


#drawMatChange(Xs, a, td, (5,10))
def drawMatChange(Xs, a, td, link, cmap=plt.cm.bwr, keep=0, binary=0):
    mat1=probX(Xs, a, td, returnmat=1)
    removed=a[link[0],link[1]]
    if removed:
        print "link removed"
    else:
        print "link added"
    a[link[0],link[1]] = 1-a[link[0],link[1]]
    a[link[1],link[0]] = 1-a[link[1],link[0]]
    mat2=probX(Xs, a, td, returnmat=1)
    if mat2 == -np.inf:
        print "bad change"
        a[link[0],link[1]] = 1-a[link[0],link[1]] # back to orig
        a[link[1],link[0]] = 1-a[link[1],link[0]]
        return
    newmat=np.array(mat2)-np.array(mat1)
    if binary:
        for inum, i in enumerate(newmat):
            for jnum, j in enumerate(i):
                if j > 0:
                    newmat[inum,jnum]=1
                else:
                    newmat[inum,jnum]=0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(newmat, interpolation='nearest', cmap=cmap, vmin=-0.03, vmax=0.03)
    plt.colorbar()
    plt.ion()
    plt.show()
    sum1=sum([sum(i) for i in mat1])
    sum2=sum([sum(i) for i in mat2])
    if sum2 > sum1:
        print "BETTER!", sum2-sum1
    else:
        print "worse :(", sum2-sum1
    if (not keep) or ((keep) and (sum1 >= sum2)):
        a[link[0],link[1]] = 1-a[link[0],link[1]] # back to orig
        a[link[1],link[0]] = 1-a[link[1],link[0]]
    return
