# morais data
# node degree <k> = 2*(numedges/numnodes)

node_degree=[2*(20224/9429.0), (2*4805)/2303.0, (2*8904)/5100.0, (2*3271)/1358.0, (2*22800)/9129.0, (2*5738)/3239.0]
cc=[.1, .2, .18, .32, .18, .24]
aspl=[5.76, 5.3, 5.84, 4.89, 5.19, 5.65]

c_rand=[.00156, .00688, .00348, .0116, .00269, .00404]
l_rand=[6.4, 5.5, 6.4, 4.69, 5.55, 5.73]

np.mean(node_degree) # average node degree ~4
np.mean(cc)          # average cc is .2
np.mean(aspl)        # aspl is 5.4

numnodes=60
numlinks=4
prob_rewire=.3


# our experiment data-- mean rt=4.9s, sd=3.9s

#n=15, m=5, std=9.5
#n=85, m=3.5, std=5.5
