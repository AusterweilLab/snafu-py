# create dicts for passing instead of passing many variables
# fill in missing variables with defaults

import warnings
from helper import *
import numpy as np

def Data(data):
    tdkeys=data.keys()

    # full factorial of any list params
    for i in tdkeys:
        if isinstance(data[i],list):
            return flatten_list([Data(dict(data, **{i: j})) for j in data[i]])

    if 'trim' not in tdkeys:
        data['trim'] = 1.0           # each list covers full graph by default
    if 'jump' not in tdkeys:
        data['jump'] = 0.0           # no jumping in data by default
    if 'jumptype' not in tdkeys:
        data['jumptype']="uniform"   # or stationary
    if 'startX' not in tdkeys:
        data['startX']="stationary"      # or stationary
    if 'numx' not in tdkeys:
        raise ValueError("Must specify 'numx' in data!")
    if 'priming' not in tdkeys:
        data['priming']=0.0
    if 'jumponcensored' not in tdkeys:
        data['jumponcensored']=None

    return dotdict(data)

def Graphs(graphs):
    tgkeys=graphs.keys()
    
    # full factorial of any list params
    for i in tgkeys:
        if isinstance(graphs[i],list):
            return flatten_list([Graphs(dict(graphs, **{i: j})) for j in graphs[i]])

    if 'numgraphs' not in tgkeys:
        graphs['numgraphs'] = 1
    if 'graphtype' not in tgkeys:
        raise ValueError("Must specify 'graphtype' in graphs!")
    if 'numnodes' not in tgkeys:
        raise ValueError("Must specify 'numnodes' in graphs!")
    if graphs['graphtype'] == "wattsstrogatz":
        if 'numlinks' not in tgkeys:
            raise ValueError("Must specify 'numlinks' in graphs!")
        if 'prob_rewire' not in tgkeys:
            raise ValueError("Must specify 'prob_rewire' in graphs!")

    return dotdict(graphs)
        
def Irts(irts):
    irtkeys=irts.keys()

    # full factorial of any list params
    for i in irtkeys:
        if (isinstance(irts[i],list)) and (i != 'data'):       # 'data' is an exception to the rule
            return flatten_list([Irts(dict(irts, **{i: j})) for j in irts[i]])

    if 'data' not in irtkeys:
        irts['data']=[]

    if 'irttype' not in irtkeys:
        if len(irts['data']) > 0:        # error unless empty dict (no IRTs)
            raise ValueError("Must specify 'irttype' in irts!")
        else:
            irts['irttype']="none"

    if 'rcutoff' not in irtkeys:
        irts['rcutoff']=20

    if irts['irttype'] == "gamma":
        if 'gamma_beta' not in irtkeys:
            irts['gamma_beta'] = (1/1.1)
            #warnings.warn("Using default beta (Gamma IRT) weight of "+str(irts['gamma_beta']))
    if irts['irttype'] == "exgauss":
        if 'exgauss_lambda' not in irtkeys:
            irts['exgauss_lambda'] = 0.5
            #warnings.warn("Using default exgauss_lambda (Ex-Gaussian IRT) weight of "+str(irts['exgauss_lambda']))
        if 'exgauss_sigma' not in irtkeys:
            irts['exgauss_sigma'] = 0.5
            #warnings.warn("Using default exgauss_sigma (Ex-Gaussian IRT) weight of "+str(irts['exgauss_sigma']))

    return dotdict(irts)

def Fitinfo(fitinfo):
    fitkeys=fitinfo.keys()

    # full factorial of any list params
    for i in fitkeys:
        if isinstance(fitinfo[i],list):
            return flatten_list([Fitinfo(dict(fitinfo, **{i: j})) for j in fitinfo[i]])
    
    if 'followtype' not in fitkeys:
        fitinfo['followtype'] = "avg"   # or max or random
    if 'prior_method' not in fitkeys:
        fitinfo['prior_method'] = "zeroinflatedbetabinomial"
    if 'zib_p' not in fitkeys:
        fitinfo['zib_p'] = 0.5
    if 'prior_a' not in fitkeys:        # adjust default prior_a depending on BB or ZIBB, to make edge prior .5
        if fitinfo['prior_method'] == "zeroinflatedbetabinomial":
            fitinfo['prior_a'] = 2
        else:
            fitinfo['prior_a'] = 1
    if 'prior_b' not in fitkeys:
        fitinfo['prior_b'] = 1
    if 'directed' not in fitkeys:
        fitinfo['directed'] = False
    if 'startGraph' not in fitkeys:
        fitinfo['startGraph'] = "goni_valid"
    if 'prune_limit' not in fitkeys:
        fitinfo['prune_limit'] = np.inf
    if 'triangle_limit' not in fitkeys:
        fitinfo['triangle_limit'] = np.inf
    if 'other_limit' not in fitkeys:
        fitinfo['other_limit'] = np.inf
    if 'goni_size' not in fitkeys:
        fitinfo['goni_size'] = 2
    if 'goni_threshold' not in fitkeys:
        fitinfo['goni_threshold'] = 2
    if 'record' not in fitkeys:
        fitinfo['record'] = False
    return dotdict(fitinfo)

