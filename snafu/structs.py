# create dicts for passing instead of passing many variables
# fill in missing variables with defaults

from . import *

class Data():
    
    def __init__(self, data):
        
        self.rawdata = data
    
        def generateGroupSpace():
            items = data['items']
            groupitems={}
            idx=0
            for subitems in items:
                for item in list(items[subitems].values()):
                    if item not in list(groupitems.values()):
                        groupitems[idx] = item
                        idx += 1
            return groupitems
        
        self.groupitems = generateGroupSpace()
        self.groupnumnodes = len(self.groupitems)
        self.subs = sorted(data['Xs'].keys())
        self.listnums = [(sub, listnum) for sub in self.subs for listnum in sorted(self.rawdata['Xs'][sub].keys())]
        
        if data['structure']:
            self.structure = "hierarchical"
            self.hierarchical()
        else:
            self.structure = "nonhierarchical"
            self.nonhierarchical()
        return

    def hierarchical(self):

        Xs = copy.deepcopy(self.rawdata['Xs'])
        self.Xs = [[Xs[i][j] for j in sorted(Xs[i].keys())] for i in self.subs]
        self.lists = self.Xs
        
        items = copy.deepcopy(self.rawdata['items'])
        self.items = [items[i] for i in sorted(items.keys())]

        self.labeledXs = [numToItemLabel(self.Xs[i],self.items[i]) for i in range(len(self.Xs))]
        self.labeledlists = self.labeledXs
        
        try:
            irts = copy.deepcopy(self.rawdata['irts'])
            self.irts = [[irts[i][j] for j in sorted(irts[i].keys())] for i in self.subs]
        except:
            self.irts = []

        self.numnodes = [len(i) for i in self.items]
        self.structure = "hierarchical"

        return self

    def nonhierarchical(self):
        # map everyone to group space
        reverseGroup = reverseDict(self.groupitems)
        Xs = copy.deepcopy(self.rawdata['Xs'])
        items = copy.deepcopy(self.rawdata['items'])
        irts = copy.deepcopy(self.rawdata['irts'])

        for sub in Xs:
            for listnum in Xs[sub]:
                Xs[sub][listnum] = [reverseGroup[items[sub][i]] for i in Xs[sub][listnum]]
                
        try:
            self.Xs = flatten_list([[Xs[i][j] for j in sorted(Xs[i].keys())] for i in self.subs])
            self.lists = self.Xs
            self.irts = flatten_list([[irts[i][j] for j in sorted(irts[i].keys())] for i in self.subs])
        except:
            self.irts = []

        self.numnodes = self.groupnumnodes
        self.items = self.groupitems
        self.structure = "nonhierarchical"
        self.labeledXs = numToItemLabel(self.Xs, self.items)
        self.labeledlists = self.labeledXs
        
        return self

    def subs(self):
        return 
    
        
def DataModel(data):
    tdkeys=list(data.keys())

    # full factorial of any list params
    for i in tdkeys:
        if isinstance(data[i],list):
            return flatten_list([Data(dict(data, **{i: j})) for j in data[i]])

    if 'trim' not in tdkeys:
        data['trim'] = 1.0           # each list covers full graph by default
    if 'jump' not in tdkeys:
        data['jump'] = 0.0           # no jumping in data by default
    if 'jumptype' not in tdkeys:
        data['jumptype'] = "uniform"   # or stationary
    if 'start_node' not in tdkeys:
        data['start_node'] = "stationary"      # or stationary
    #if 'numx' not in tdkeys:
    #    raise ValueError("Must specify 'numx' in data!")
    if 'priming' not in tdkeys:
        data['priming'] = 0.0
    if 'jumponcensored' not in tdkeys:
        data['jumponcensored'] = None
    if 'censor_fault' not in tdkeys:    # for generative process only, not inferring graphs
        data['censor_fault'] = 0.0
    if 'emission_fault' not in tdkeys:  # for generative process only, not inferring graphs
        data['emission_fault'] = 0.0
    
    return dotdict(data)

def Irts(irts):
    irtkeys=list(irts.keys())

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
    fitkeys=list(fitinfo.keys())

    # full factorial of any list params
    for i in fitkeys:
        if isinstance(fitinfo[i],list):
            return flatten_list([Fitinfo(dict(fitinfo, **{i: j})) for j in fitinfo[i]])
    
    if 'followtype' not in fitkeys:
        fitinfo['followtype'] = "avg"   # or max or random
    if 'prior_method' not in fitkeys:
        fitinfo['prior_method'] = "zeroinflatedbetabinomial"
    if 'zibb_p' not in fitkeys:
        fitinfo['zibb_p'] = 0.5
    if 'prior_b' not in fitkeys:
        fitinfo['prior_b'] = 1
    if 'prior_a' not in fitkeys:        # adjust default prior_a depending on BB or ZIBB, to make edge prior .5
        if fitinfo['prior_method'] == "betabinomial":
            fitinfo['prior_a'] = fitinfo['prior_b']
        if fitinfo['prior_method'] == "zeroinflatedbetabinomial":
            fitinfo['prior_a'] = fitinfo['prior_b'] / float(fitinfo['zibb_p'])
    if 'directed' not in fitkeys:
        fitinfo['directed'] = False
    if 'startGraph' not in fitkeys:
        fitinfo['startGraph'] = "cn_valid"
    if 'prune_limit' not in fitkeys:
        fitinfo['prune_limit'] = np.inf
    if 'triangle_limit' not in fitkeys:
        fitinfo['triangle_limit'] = np.inf
    if 'other_limit' not in fitkeys:
        fitinfo['other_limit'] = np.inf
    if 'cn_windowsize' not in fitkeys:
        fitinfo['cn_windowsize'] = 2
    if 'cn_threshold' not in fitkeys:
        fitinfo['cn_threshold'] = 2
    if 'cn_alpha' not in fitkeys:
        fitinfo['cn_alpha'] = 0.05

    return dotdict(fitinfo)

