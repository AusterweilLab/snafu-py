def distort(graph):
    newgraph=np.copy(graph)
    zeros=zip(*np.where(graph==0))
    ones=zip(*np.where(graph==1))
    changetoone=np.random.choice(range(len(zeros)),39,replace=False)
    changetozero=np.random.choice(range(len(ones)),39,replace=False)
   
    for i in changetoone:
        idx=zeros[i]
        idx2=zeros[i][::-1]
        graph[idx]=1
        graph[idx2]=1
   
    for i in changetozero:
        idx=ones[i]
        idx2=ones[i][::-1]
        graph[idx]=0
        graph[idx2]=0
   
   return newgraph
