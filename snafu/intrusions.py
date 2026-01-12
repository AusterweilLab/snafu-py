import numpy as np
from . import *

def intrusionsList(l, scheme):  
    """
    Identify intrusions in fluency lists based on a target scheme.

    Parameters
    ----------
    l : list
        A list of lists representing recalled items. Can be hierarchical (list of lists of lists).
    scheme : list or str
        Target category or categories used to identify intrusions. Can be a single category (str)
        or a list specifying valid categories.

    Returns
    -------
    list
        A list of intruded items for each list or participant. Maintains original hierarchical structure."""      
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
    """
    Compute number of intrusions per list or per participant.

    Parameters
    ----------
    l : list
        Fluency data, either a list of lists (non-hierarchical) or a list of list of lists (hierarchical).
    scheme : list or str
        Target category or categories used to identify intrusions.

    Returns
    -------
    list of float
        Number of intrusions per list (non-hierarchical) or mean intrusions per participant (hierarchical).
    """    
    ilist = intrusionsList(l, scheme)
    
    # if fluency data are hierarchical, report mean per individual
    if isinstance(l[0][0], list):
        return [np.mean([len(i) for i in subj]) for subj in ilist]
    # if fluency data are non-hierarchical, report mean per list
    else:
        return [float(len(i)) for i in ilist]
