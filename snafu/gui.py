from . import *

def list_subjects_and_categories(command, root_path):
    subjects=[]
    categories=[]
    groups=["all"]
    
    with open(command['fullpath'],'rt',encoding="utf-8-sig") as fh:
        header=fh.readline().strip().split(',')
        subj_idx = header.index("id")
        cat_idx = header.index("category")
        try:
            group_idx = header.index("group")
        except:
            group_idx = -1

        from csv import reader
        for line in reader(fh):
            if line[subj_idx] not in subjects:
                subjects.append(line[subj_idx])
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
   
    # turns array into string: "Mean (Std) [Min - Max]"
    def format_output(x):
        x_mean = str(round(np.mean(x),2))
        x_std = str(round(np.std(x),2))
        x_min = str(round(np.min(x),2))
        x_max = str(round(np.max(x),2))
        
        return x_mean + " (" + x_std + ") [" + x_min + " -- " + x_max + "]"
    
    command = command['data_parameters']
  
    if command['factor_type'] == "subject":
        subject = str(command['subject'])
        group = None
    elif command['factor_type'] == "group":
        if command['group'] != "all":
            group = command['group']
        else:
            group = None                             # reserved group label in GUI for all subjects
        subject = None

    filedata = load_fluency_data(command['fullpath'], category=command['category'], spell=label_to_filepath(command['spellfile'], root_path, "spellfiles"), group=group, subject=subject, hierarchical=True)
    Xs = filedata.Xs
    labeledXs = filedata.labeledXs
    items = filedata.items
    irts = filedata.irts
    numnodes = filedata.numnodes

    # initialize
    avg_cluster_size = ["n/a"]
    avg_num_cluster_switches = ["n/a"]
    avg_num_intrusions = ["n/a"]
    num_lists = []
    avg_items_listed = []
    avg_unique_items_listed = []
    list_of_intrusions = []
    avg_num_perseverations = []
    list_of_perseverations = []
   
    if not command['freq_ignore']:
        try:
            freq_sub = float(command['freq_sub'])
        except:
            freq_sub = None
    else:
        freq_sub = None

    if not command['aoa_ignore']:
        try:
            aoa_sub = float(command['aoa_sub'])
        except:
            aoa_sub = None
    else:
        aoa_sub = None


    freqfile = label_to_filepath(command['freqfile'], root_path,"frequency")
    aoafile = label_to_filepath(command['aoafile'], root_path,"aoa")
    
    preset_schemes = {"Phonemic: 1 letter": 1,
                      "Phonemic: 2 letters": 2,
                      "Phonemic: 3 letters": 3}
    if command['cluster_scheme'] in preset_schemes.keys():
        schemefile = preset_schemes[command['cluster_scheme']]
    else:
        schemefile = label_to_filepath(command['cluster_scheme'], root_path, "schemes")
    
    total_words = 0
    avg_word_freq = []
    word_freq_excluded = []
    avg_word_aoa = []
    word_aoa_excluded = []

    if command['cluster_scheme'] != "None":
        avg_cluster_size = clusterSize(labeledXs, schemefile, clustertype=command['cluster_type'])
        avg_num_cluster_switches = clusterSwitch(labeledXs, schemefile, clustertype=command['cluster_type'])
        avg_num_intrusions = intrusions(labeledXs, schemefile)
        list_of_intrusions = intrusionsList(labeledXs, schemefile)
    avg_num_perseverations = perseverations(labeledXs)
    list_of_perseverations = perseverationsList(labeledXs)

    for subjnum in range(len(labeledXs)):
        num_lists.append(len(labeledXs[subjnum]))
        avg_items_listed.append(np.mean([len(i) for i in labeledXs[subjnum]]))
        avg_unique_items_listed.append(np.mean([len(set(i)) for i in labeledXs[subjnum]]))

        freq, excluded = wordFrequency(labeledXs[subjnum],missing=freq_sub,data=freqfile)
        avg_word_freq.append(freq)
        for word in excluded:
            word_freq_excluded.append(word)
        aoa, excluded = ageOfAcquisition(labeledXs[subjnum], missing=aoa_sub, data=aoafile)
        avg_word_aoa.append(aoa)
        for word in excluded:
            word_aoa_excluded.append(word)
        for i in labeledXs[subjnum]:
            total_words += len(i)

    # clean up / format data to send back, still messy
    list_of_intrusions = flatten_list(list_of_intrusions)
    list_of_perseverations = flatten_list(list_of_perseverations)

    if len(labeledXs) > 1:
        if command['cluster_scheme'] != "None":
            avg_cluster_size = format_output(avg_cluster_size)
            avg_num_cluster_switches = format_output(avg_num_cluster_switches)
            avg_num_intrusions = format_output(avg_num_intrusions)
            avg_num_perseverations = format_output(avg_num_perseverations)
        num_lists = format_output(num_lists)
        avg_items_listed = format_output(avg_items_listed)
        avg_unique_items_listed = format_output(avg_unique_items_listed)
        avg_word_freq=format_output(avg_word_freq)
        avg_word_aoa=format_output(avg_word_aoa)
 
    word_freq_rate = 0
    for i in word_freq_excluded:
        word_freq_rate += len(i)
    word_freq_rate = str(round(float(word_freq_rate)/total_words*100,2))+'%'
    word_aoa_rate = 0
    for i in word_aoa_excluded:
        word_aoa_rate += len(i)
    word_aoa_rate = str(round(float(word_aoa_rate)/total_words*100,2))+'%'
    # fix
    csv_file = generate_csv_file(command, root_path)

    return { "type": "data_properties", 
             "num_lists": num_lists,
             "avg_items_listed": avg_items_listed,
             "intrusions": list_of_intrusions,
             "perseverations": list_of_perseverations,
             "avg_num_intrusions": avg_num_intrusions,
             "avg_num_perseverations": avg_num_perseverations,
             "avg_unique_items_listed": avg_unique_items_listed,
             "avg_num_cluster_switches": avg_num_cluster_switches,
             "avg_cluster_size": avg_cluster_size,
             "avg_word_freq": avg_word_freq,
             "avg_word_aoa": avg_word_aoa,
             "word_freq_rate": word_freq_rate,
             "word_freq_excluded": word_freq_excluded,
             "word_aoa_rate": word_aoa_rate,
             "word_aoa_excluded": word_aoa_excluded,
             "csv_file": csv_file }

# broken
def generate_csv_file(command, root_path):
    return ### temporary, function is broken JZ
    csv_file = "id,listnum,num_items_listed,num_unique_items,num_cluster_switches,avg_cluster_size,num_intrusions,num_perseverations,avg_word_freq,avg_word_aoa\n"
    filedata = load_fluency_data(command['fullpath'],category=command['category'], scheme=label_to_filepath(command['cluster_scheme'], root_path, "schemes"), spellfile=label_to_filepath(command['spellfile'], root_path, "spellfiles"))
    filedata.hierarchical()
    
    for subnum, sub in enumerate(filedata.subs):
        labeledXs = filedata.labeledXs[subnum]
        for listnum in range(len(filedata.Xs[subnum])):
            csv_sub = sub
            csv_listnum = listnum
            csv_numitems = len(filedata.Xs[subnum][listnum])
            csv_uniqueitem = len(set(filedata.Xs[subnum][listnum]))
            
            # parameters should come from snafu gui (scheme, clustertype)
            csv_clusterlength = clusterSize(labeledXs, scheme=label_to_filepath(command['cluster_scheme'], root_path, "schemes"), clustertype=command['cluster_type'])
            csv_clusterswitch = clusterSwitch(labeledXs, scheme=label_to_filepath(command['cluster_scheme'], root_path, "schemes"), clustertype=command['cluster_type'])

            # parameters should come from snafu gui (scheme)
            csv_intrusions = intrusions(labeledXs,scheme=label_to_filepath(command['cluster_scheme'], root_path, "schemes"))
            csv_perseverations = perseverations(labeledXs)

            csv_freq, temp = wordFrequency([labeledXs[listnum]],freq_sub=float(command['freq_sub']))
            csv_aoa, temp = ageOfAcquisition([labeledXs[listnum]])
            csv_freq = np.mean(csv_freq)
            csv_aoa = np.mean(csv_aoa)

            csv_file += str(csv_sub)+','+str(csv_listnum)+','+str(csv_numitems)+','+str(csv_uniqueitem)+','+str(csv_clusterswitch)+','+str(round(csv_clusterlength,2))+','+str(csv_intrusions)+','+str(csv_perseverations)+','+str(round(csv_freq,2))+','+str(round(csv_aoa,2))+'\n'

    return csv_file


def network_properties(command, root_path):
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
    irts = filedata.irts
    numnodes = filedata.numnodes
    
    toydata=DataModel({
            'numx': len(Xs),
            'trim': 1,
            'jump': float(command['jump_probability']),
            'jumptype': command['jump_type'],
            'priming': float(command['priming_probability']),
            'start_node': command['first_item']})
    fitinfo=Fitinfo({
            'prior_method': "zeroinflatedbetabinomial",
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
    elif command['network_method']=="Community Network":
        bestgraph = communityNetwork(Xs, fitinfo=fitinfo, numnodes=numnodes)
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
    nxg_json = json.load(open(command['fullpath'],'rt',encoding="utf-8-sig"))
    nxg = nx.readwrite.nx.readwrite.json_graph.node_link_graph(
        nxg_json,
        multigraph = False,
        attrs=dict(source='source', target='target', name='id', key='nodes', link='edges')
    )
    return graph_properties(nxg, nxg_json)

def graph_properties(nxg,nxg_json): # separate function that calculates graph properties
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
    return { "type": "quit",
             "status": "success" }

def error(msg):
    return { "type": "error",
             "msg": msg }
