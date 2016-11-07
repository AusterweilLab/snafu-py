import rw

outfile='tmp2.csv'
header=1

toygraphs=rw.Toygraphs({
        'numgraphs': 10,
        'graphtype': "steyvers",
        'numnodes': 160,
        'numlinks': 6,
        'prob_rewire': .3})

toydata=rw.Toydata({
        'numx': range(3,15),
        'trim': .7,
        'jump': 0.0,
        'jumptype': "uniform",
        'start': "stationary"})

irts=rw.Irts({
        'data': [],
        'irttype': "gamma",
        'beta': (1/1.1),
        'irt_weight': 0.9,
        'rcutoff': 20})

fitinfo=rw.Fitinfo({
        'tolerance': 1500,
        'start': "naiverw",
        'prob_multi': .8,
        'prob_overlap': .8})

# optionally, pass a methods argument
# default is methods=['fe','rw','uinvite','uinvite_irt'] 
for td in toydata:
    rw.toyBatch(toygraphs, toydata, outfile, irts=irts, start_seed=1,methods=['rw','uinvite','uinvite_prior','uinvite_irt','uinvite_irt_prior'],header=header,debug="T")
    header=0
