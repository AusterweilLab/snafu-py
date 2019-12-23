from . import *

def list_subjects_and_categories(command, root_path):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    subjects=[]
    #categories=["all"]
    categories = []
    groups=["all"]
    
    with open(command['fullpath'],'rt',encoding="utf-8-sig") as fh:
        header=fh.readline().strip().split(',')
        subj_idx = header.index("id")
        try:
            cat_idx = header.index("category")
        except:
            cat_idx = -1
        try:
            group_idx = header.index("group")
        except:
            group_idx = -1

        from csv import reader
        for line in reader(fh):
            if line[subj_idx] not in subjects:
                subjects.append(line[subj_idx])
            if cat_idx > -1:
                if line[cat_idx] not in categories:
                    categories.append(line[cat_idx])
            if group_idx > -1:
                group_label = line[group_idx]
                if group_label not in groups:
                    groups.append(group_label)
    return { "type": "list_subjects_and_categories",
             "subjects": subjects, 
             "categories": categories,
             "groups": groups,
             "subject": subjects[0],
             "category": categories[0],
             "group": groups[0] }

def jsonGraph(g, items):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    json_data = nx.readwrite.json_graph.node_link_data(g)
    
    json_data['edges'] = json_data['links']
    json_data.pop('links', None)
    json_data.pop('directed', None)
    json_data.pop('multigraph', None)
    json_data.pop('graph', None)
    
    for i, j in enumerate(json_data['edges']):
        json_data['edges'][i]['id'] = i
        
        # this line fixes a bug (JSON not serializable) i don't know why it's not serializable without it (py3.5)
        json_data['edges'][i]['target'] = int(json_data['edges'][i]['target']) 
   
    for i, j in enumerate(json_data['nodes']):
        json_data['nodes'][i]['label'] = items[j['id']]
   
    # works when edges['target'] aren't in dict
    #json_data.pop('edges',None)

    return json_data

def label_to_filepath(x, root_path, filetype):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    filedict=dict()
    folder = root_path + "/" + filetype + "/"

    # since filenames are populated from directory listing there's no need to build a dict() anymore, but that's the way it's done still
    for filename in os.listdir(folder):
        if "csv" in filename:
            label = filename[0:filename.find('.')].replace('_',' ')
            filedict[label] = folder + filename
    try:
        filename = filedict[x]
    except:
        filename = None
    return filename

def data_properties(command, root_path):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
   
    # data parameters shorthand
    command = command['data_parameters']

    # Initialize some filenames
    freqfile = label_to_filepath(command['freqfile'], root_path, "frequency")
    aoafile = label_to_filepath(command['aoafile'], root_path, "aoa")
    
    # Letter fluency scheme or semantic fluency scheme?
    preset_schemes = {"1 letter": 1,
                      "2 letters": 2,
                      "3 letters": 3}
    if command['cluster_scheme'] in preset_schemes.keys():
        schemefile = preset_schemes[command['cluster_scheme']]
    else:
        schemefile = label_to_filepath(command['cluster_scheme'], root_path, "schemes")
    
    # Import a single subject or group of subjects?
    if command['factor_type'] == "subject":
        subject = str(command['subject'])
        group = None
    elif command['factor_type'] == "group":
        if command['group'] != "all":
            group = command['group']
        else:
            group = None
        subject = None
    if command['category'] == "all":
        category = None
    else:
        category = command['category']

    # Load fluency data
    filedata = load_fluency_data(command['fullpath'], category=category, spell=label_to_filepath(command['spellfile'], root_path, "spellfiles"), group=group, subject=subject, hierarchical=True)
    labeledXs = filedata.labeledXs

    # Set imputation value for word frequency
    if not command['freq_ignore']:
        try: freq_sub = float(command['freq_sub'])
        except: return error("Missing frequency value must be a number")
    else: freq_sub = None

    # Set imputation value for age-of-acquisition
    if not command['aoa_ignore']:
        try: aoa_sub = float(command['aoa_sub'])
        except: return error("Missing age-of-acquisition value must be a number")
    else: aoa_sub = None

    # List of perseverations and calculate average perseverations per participant
    list_of_perseverations = perseverationsList(labeledXs)
    avg_num_perseverations_list = [[len(l) for l in subj] for subj in list_of_perseverations]
    avg_num_perseverations = np.mean([np.mean(i) for i in avg_num_perseverations_list])
    avg_num_perseverations_list = flatten_list(avg_num_perseverations_list)
    list_of_perseverations = flatten_list(list_of_perseverations)
    
    if command['cluster_scheme'] != "None":
        # calculate cluster sizes
        list_of_clusters = findClusters(labeledXs, schemefile, clustertype=command['cluster_type'])
        avg_cluster_size_list = [[np.mean(l) for l in subj] for subj in list_of_clusters]
        avg_cluster_size = np.mean([np.mean(i) for i in avg_cluster_size_list])
        avg_cluster_size_list = flatten_list(avg_cluster_size_list)
        
        # calculate cluster switches
        avg_num_cluster_switches_list = [[len(l)-1 for l in subj] for subj in list_of_clusters]
        avg_num_cluster_switches = np.mean([np.mean(i) for i in avg_num_cluster_switches_list])
        avg_num_cluster_switches_list = flatten_list(avg_num_cluster_switches_list)
        
        # calculate intrusions
        if command['fluency_type'] == "semantic":
            list_of_intrusions = intrusionsList(labeledXs, schemefile)
        elif command['fluency_type'] == "letter":
            list_of_intrusions = intrusionsList(labeledXs, command['target_letter'])
        avg_num_intrusions_list = [[len(l) for l in subj] for subj in list_of_intrusions]
        avg_num_intrusions = np.mean([np.mean(i) for i in avg_num_intrusions_list])
        avg_num_intrusions_list = flatten_list(avg_num_intrusions_list)
        list_of_intrusions = flatten_list(list_of_intrusions)
    else:
        # if no cluster scheme is provided, set defaults for cluster size/switches and intrusions
        list_of_intrusions = []
        avg_num_intrusions = "n/a"
        avg_cluster_size = "n/a"
        avg_num_cluster_switches = "n/a"
        # arbitrarily use `avg_num_perseverations_list` to create empty lists with correct number of elements
        avg_num_intrusions_list = ["" for i in range(len(avg_num_perseverations_list))]
        avg_num_cluster_switches_list = ["" for i in range(len(avg_num_perseverations_list))]
        avg_cluster_size_list = ["" for i in range(len(avg_num_perseverations_list))]

    # count number of lists per participant
    num_lists_list = [len(subj) for subj in labeledXs]
    num_lists = np.mean(num_lists_list)
  
    # number of items listed for each participant
    avg_items_listed_list = [[len(l) for l in subj] for subj in labeledXs]
    avg_items_listed = np.mean([np.mean(i) for i in avg_items_listed_list])
    avg_items_listed_list = flatten_list(avg_items_listed_list)
  
    # calculate age-of-acquisition and word frequency
    # loads word lists for each subject which could be optimized...
    aoa_list = []
    word_aoa_excluded = []
    freq_list = []
    word_freq_excluded = []
    
    for subj in labeledXs:
        aoa, excluded = wordStat(subj, missing=aoa_sub, data=aoafile)
        aoa_list.append(aoa)
        word_aoa_excluded.append(excluded)

        freq, excluded = wordStat(subj, missing=freq_sub, data=freqfile)
        freq_list.append(freq)
        word_freq_excluded.append(excluded)
        
    avg_word_freq = np.mean([np.mean(i) for i in freq_list])
    avg_word_aoa = np.mean([np.mean(i) for i in aoa_list])
    freq_list = flatten_list(freq_list)
    aoa_list = flatten_list(aoa_list)
    word_freq_excluded = flatten_list(word_freq_excluded)
    word_aoa_excluded = flatten_list(word_aoa_excluded)
 
    # calculate % of missing word tokens from age-of-acquisition and word frequency dictionaries
    total_words = len(flatten_list(labeledXs,2))
    word_freq_rate = len(flatten_list(word_freq_excluded)) / float(total_words)
    word_freq_rate = str(round(word_freq_rate * 100, 2))+'%'
    word_aoa_rate = len(flatten_list(word_aoa_excluded)) / float(total_words)
    word_aoa_rate = str(round(word_aoa_rate * 100, 2))+'%'
   
    # list of subjects and list nums
    subsandlists = list(zip(*filedata.listnums))
    subs = list(subsandlists[0])
    listnums = list(subsandlists[1])
    
    # make csv file
    from collections import OrderedDict
    csv_data = OrderedDict()
    csv_data["subject"] = subs
    csv_data["list_number"] = listnums
    csv_data["num_responses"] = avg_items_listed_list
    csv_data["num_intrusions"] = avg_num_intrusions_list
    csv_data["num_perseverations"] = avg_num_perseverations_list
    csv_data["num_cluster_switches"] = avg_num_cluster_switches_list
    csv_data["avg_cluster_size"] = avg_cluster_size_list
    csv_data["avg_word_freq"] = freq_list
    csv_data["avg_word_aoa"] = aoa_list
    csv_file = generate_csv_file(csv_data)

    # list of spelling corrections
    spell_corrected = flatten_list(filedata.spell_corrected)
    spell_corrected = [[list(i) for i in l] for l in spell_corrected]
    num_spell_corrections = sum([len(l) for l in spell_corrected])
    
    return { "type": "data_properties", 
             "listnums": filedata.listnums,
             "num_lists": num_lists,
             "avg_items_listed": avg_items_listed,
             "intrusions": list_of_intrusions,
             "perseverations": list_of_perseverations,
             "avg_num_intrusions": avg_num_intrusions,
             "avg_num_perseverations": avg_num_perseverations,
             "avg_num_cluster_switches": avg_num_cluster_switches,
             "avg_cluster_size": avg_cluster_size,
             "avg_word_freq": avg_word_freq,
             "avg_word_aoa": avg_word_aoa,
             "word_freq_rate": word_freq_rate,
             "word_freq_excluded": word_freq_excluded,
             "word_aoa_rate": word_aoa_rate,
             "word_aoa_excluded": word_aoa_excluded,
             "spell_corrected": spell_corrected,
             "num_spell_corrections": num_spell_corrections,
             "csv_file": csv_file }

def generate_csv_file(json_file):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    keys = list(json_file.keys())
    header_row = ",".join(keys)

    csv_file = header_row + "\n"
    
    to_write = [",".join([str(j) for j in i])+"\n" for i in list(zip(*list(json_file.values())))]
    for line in to_write:
        csv_file += line
    
    return csv_file



def network_properties(command, root_path):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    subj_props = command['data_parameters']
    command = command['network_parameters']

    # U-INVITE won't work with perseverations
    if command['network_method'] == "U-INVITE":
        removePerseverations=True
    else:
        removePerseverations=False
   
    if subj_props['factor_type'] == "subject":
        subject = str(subj_props['subject'])
        group = None
    elif subj_props['factor_type'] == "group":
        if subj_props['group'] != "all":
            group = subj_props['group']
        else:
            group = None                             # reserved group label in GUI for all subjects
        subject = None

    filedata = load_fluency_data(subj_props['fullpath'], category=subj_props['category'], spell=label_to_filepath(subj_props['spellfile'], root_path, "spellfiles"), removePerseverations=removePerseverations, subject=subject, group=group)
    Xs = filedata.Xs
    items = filedata.items
    numnodes = filedata.numnodes
    
    toydata=DataModel({
            'numx': len(Xs),
            'trim': 1,
            'jump': float(command['jump_probability']),
            'jumptype': command['jump_type'],
            'priming': float(command['priming_probability']),
            'start_node': command['first_item']})
    fitinfo=Fitinfo({
            #'prior_method': "zeroinflatedbetabinomial",
            'prior_a': 1,
            'prior_b': 2,
            'zibb_p': 0.5,
            'startGraph': command['starting_graph'],
            'cn_windowsize': int(command['cn_windowsize']),
            'cn_threshold': int(command['cn_threshold']),
            'cn_alpha': float(command['cn_alpha']),
            'followtype': "avg", 
            'prune_limit': 100,
            'triangle_limit': 100,
            'other_limit': 100})
   
    if command['prior']=="None":
        prior=None
    elif command['prior']=="USF":
        usf_file_path = "/snet/USF_animal_subset.snet"
        filename = root_path + usf_file_path
        
        usf_graph, usf_items = read_graph(filename)
        usf_numnodes = len(usf_items)
        priordict = genGraphPrior([usf_graph], [usf_items], fitinfo=fitinfo)
        prior = (priordict, usf_items)
        
    if command['network_method']=="Naive Random Walk":
        bestgraph = naiveRandomWalk(Xs, numnodes=numnodes)
    elif command['network_method']=="Conceptual Network":
        bestgraph = conceptualNetwork(Xs, fitinfo=fitinfo, numnodes=numnodes)
    elif command['network_method']=="Pathfinder":
        bestgraph = pathfinder(Xs, numnodes=numnodes)
    elif command['network_method']=="Correlation-based Network":
        bestgraph = correlationBasedNetwork(Xs, numnodes=numnodes)
    elif command['network_method']=="First Edge":
        bestgraph = firstEdge(Xs, numnodes=numnodes)
    elif command['network_method']=="U-INVITE":
        bestgraph, ll = uinvite(Xs, toydata, numnodes=numnodes, fitinfo=fitinfo, debug=False, prior=prior)
    
    nxg = nx.to_networkx_graph(bestgraph)
    nxg_json = jsonGraph(nxg, items)
    
    return graph_properties(nxg,nxg_json)

def analyze_graph(command, root_path): # used when importing graphs
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    nxg_json = json.load(open(command['fullpath'],'rt',encoding="utf-8-sig"))
    nxg = nx.readwrite.json_graph.node_link_graph(
        nxg_json,
        multigraph = False,
        attrs=dict(source='source', target='target', name='id', key='nodes', link='edges')
    )
    return graph_properties(nxg, nxg_json)

def graph_properties(nxg,nxg_json): # separate function that calculates graph properties
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    node_degree = np.mean(list(dict(nxg.degree()).values()))
    clustering_coefficient = nx.average_clustering(nxg)
    try:
        aspl = nx.average_shortest_path_length(nxg)
    except:
        aspl = "disjointed graph"
    density = nx.classes.function.density(nxg)
    betweenness_centrality_nodes = nx.algorithms.centrality.betweenness_centrality(nxg)
    avg_betweenness_centrality = 0.0
    for node, bc in list(betweenness_centrality_nodes.items()):
        avg_betweenness_centrality += bc
    avg_betweenness_centrality /= len(betweenness_centrality_nodes)

    return {
        "type": "network_properties",
        "node_degree": node_degree,
        "clustering_coefficient": clustering_coefficient,
        "aspl": aspl,
        "density": density,
        "avg_betweenness_centrality": avg_betweenness_centrality,
        "graph": nxg_json
    }

def quit(command, root_path): 
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    return { "type": "quit",
             "status": "success" }

def error(msg):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    return { "type": "error",
             "msg": str(msg) }
