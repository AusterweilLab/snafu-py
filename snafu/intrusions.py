import numpy as np
from . import *

def intrusionsList(l, scheme):  
    if len(l) > 0:
        if isinstance(l[0][0], list):
            intrusion_items = [intrusionsList(i, scheme) for i in l]
        else:
            if len(scheme) == 1:
                labels = labelClusters(l, 1, labelIntrusions=True, targetLetter=scheme)
            else:
                labels = labelClusters(l, scheme, labelIntrusions=True)
            intrusion_items = [[l[listnum][i] for i, j in enumerate(eachlist) if j=="intrusion"] for listnum, eachlist in enumerate(labels)]
    else:
        intrusion_items = []
    return intrusion_items

def intrusions(l, scheme):
    ilist = intrusionsList(l, scheme)
    
    # if fluency data are hierarchical, report mean per individual
    if isinstance(l[0][0], list):
        return [np.mean([len(i) for i in subj]) for subj in ilist]
    # if fluency data are non-hierarchical, report mean per list
    else:
        return [float(len(i)) for i in ilist]

