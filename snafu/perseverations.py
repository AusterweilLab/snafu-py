import numpy as np
from . import *

def perseverationsList(l):
    """
    Identify repeated (perseverated) items in fluency lists.

    Handles both hierarchical (list of lists of lists) and flat (list of lists) formats.
    For each list, returns the unique items that occur more than once.

    Parameters
    ----------
    l : list
        A list of lists (or list of list of lists) representing fluency responses.

    Returns
    -------
    list
        A list of lists where each inner list contains the items repeated
        in the corresponding fluency list.
    """
    if len(l) > 0:
        if isinstance(l[0][0], list):
            perseveration_items = [perseverationsList(i) for i in l]
        else:
            perseveration_items = [list(set([item for item in ls if ls.count(item) > 1])) for ls in l]
    else:
        perseveration_items = []
    return perseveration_items


def perseverations(l):
    """
    Count the number of perseverations (repeated items) in fluency data.

    Computes the number of repeated responses per list. If the input is 
    hierarchical (e.g., per subject with multiple lists), the function 
    returns the mean number of perseverations per subject.

    Parameters
    ----------
    l : list
        Fluency data: a list of lists (or list of list of lists) of responses.

    Returns
    -------
    list of float or list of list of float
        A list of perseveration counts per list, or mean per subject if hierarchical.
    """
    def processList(l2):
        return [float(len(i)-len(set(i))) for i in l2]
    
    # if fluency data are hierarchical, report mean per individual
    if isinstance(l[0][0],list):
        return [np.mean(processList(subj)) for subj in l]
    # if fluency data are non-hierarchical, report mean per list
    else:
        return processList(l)
