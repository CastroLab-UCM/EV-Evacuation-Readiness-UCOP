import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import json
import math
import pandas as pd # type: ignore
import config # type: ignore
from haversine import haversine, Unit # type: ignore
import networkx as nx # type: ignore
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "GeoJson"

df = pd.read_csv(DATA_DIR / 'LinkFlowStatistics_NormalFlow_v20250916a.txt')

with open(DATA_DIR / 'nodes.geojson', 'r') as f:
    nodes_data = json.load(f)


with open(DATA_DIR / 'sections.geojson', 'r') as f:
    edges_data = json.load(f)

with open(DATA_DIR / 'centroids.geojson', 'r') as f:
    centroids = json.load(f)

with open(DATA_DIR / 'centroid_connections.geojson', 'r') as f:
    centroid_connections = json.load(f)

with open(DATA_DIR / 'turnings.geojson', 'r') as f:
    turns = json.load(f)

######################## extract background traffic #############################################################
#################################################################################################################
#################################################################################################################

# Standardize column names (in case of trailing spaces)
df.columns = [col.strip() for col in df.columns]

# Group by Section ID and extract the 2nd largest Flow value
second_largest_flows = (
    df.groupby("Section ID")["Flow (veh/hr)"]
        .apply(lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else None)
        .reset_index()
)

# Convert result to list of tuples (Section ID, second largest flow)
background_traffic_flow = list(second_largest_flows.itertuples(index=False, name=None))

# remove duplicate section ID with lowest flow values
unique_dict = {}
for sec_id, flow in background_traffic_flow:
    if sec_id not in unique_dict or flow > unique_dict[sec_id]:
        unique_dict[sec_id] = flow
background_flow_list = list(unique_dict.items())

######### Extracting OD pairs ###################################################################################
#################################################################################################################
#################################################################################################################

start_node = []
goal_node = []

for start,goal in zip(config.start_node_centroid, config.goal_node_centroid):

    # To obtain the nearest node in the network connected to the start centroid

    desired_start_section = next((feature["properties"].get("id_object") for feature in centroid_connections["features"] if feature["properties"].get("id_cent") == start 
                                and feature["properties"].get("direction") == "to"),None)
    start_node.append(next((feature["properties"].get("tnode") for feature in edges_data["features"] if feature["properties"].get("id") == desired_start_section ),None))

    # To obtain the nearest node in the network connected to the goal centroid

    desired_goal_section = next((feature["properties"].get("id_object") for feature in centroid_connections["features"] if feature["properties"].get("id_cent") == goal 
                                and feature["properties"].get("direction") == "from"),None)
    goal_node.append(next((feature["properties"].get("fnode") for feature in edges_data["features"] if feature["properties"].get("id") == desired_goal_section ),None))

start_node =np.array(start_node)
goal_node =np.array(goal_node)

######### Extracting nodes and links ##############################################################################
###################################################################################################################
###################################################################################################################

pos_nodes = {}
for location in nodes_data["features"]:

    pos_nodes[location["properties"]["id"]] = tuple(location["geometry"]["coordinates"][:-1])



basic_edges_temp = []
link_number_temp = []
link_speed_temp = []
link_capacity_temp = []

for location in edges_data["features"]:

    if "fnode" not in location["properties"]:
        continue
    elif "tnode" not in location["properties"]:
        continue
    else:
        basic_edges_temp.append((location["properties"]["fnode"],location["properties"]["tnode"]))

        link_number_temp.append(location["properties"]["id"])
        link_speed_temp.append(location["properties"]["speed"])

        section_background_flow = next((flow for sec_id, flow in background_flow_list if sec_id == location["properties"]["id"]),None)
        link_capacity_temp.append(location["properties"]["capacity"] - 0*section_background_flow)
        # link_capacity_temp.append(400)

seen = set()
basic_edges = []
link_number = []
link_speed = []
link_capacity = []

# # to get rid of duplicate edges
for iter, t in enumerate(basic_edges_temp):
    if t not in seen:
        basic_edges.append(t)  # Append original tuple
        seen.add(t)  # Track seen tuples

        link_number.append(link_number_temp[iter])
        link_speed.append(link_speed_temp[iter])
        link_capacity.append(link_capacity_temp[iter])


link_number = np.array(link_number)
link_speed = np.array(link_speed)
link_capacity = np.array(link_capacity)

# print(link_speed.mean())
# print(link_speed.min())
# print(link_speed.max())



num_nodes = len(pos_nodes)
num_links = len(basic_edges)

road_dist = np.zeros(num_links)
road_travel_time = np.zeros(num_links)

edge_information = []
for i in range(num_links):
    road_dist[i] = haversine(pos_nodes[basic_edges[i][0]][::-1],pos_nodes[basic_edges[i][1]][::-1],unit=Unit.KILOMETERS)
    road_travel_time[i] = (1/link_speed[i])*road_dist[i]
    edge_information.append((basic_edges[i][0],basic_edges[i][1],{'travel_time':road_travel_time[i],
                                                                  'road_limit':link_capacity[i],
                                                                  'travel_dist':road_dist[i]}))

#################### Information about FCS and MCS #############################################################
################################################################################################################
################################################################################################################
################################################################################################################

FCS_charging_time = config.stoppage_time
FCS_charged_dist = np.zeros(num_links)
FCS_loc = np.zeros(num_links)
FCS_flow_limit_vector = np.zeros(num_links)
max_EV_FCS = np.zeros(num_links)

MCS_charging_time = config.stoppage_time
MCS_charged_dist = config.MCS_range_per_hour*MCS_charging_time
MCS_loc = np.zeros(num_links)
max_EV_MCS = np.zeros(num_links)

FCS_count = 0
MCS_count = 0
FCS_ignored = [2,3,4]
MCS_ignored = []#[5,6,15]

for location in  centroids["features"]:

    if location["properties"]["name"] != "":

        target_id1 = location["properties"]["id"]


        desired_dict3_list = [feature for feature in centroid_connections["features"] if feature["properties"].get("id_cent") == target_id1
                              and feature["properties"].get("direction") == "from"]
        
        FCS_check = 0
        MCS_check = 0

        for i in range(len(desired_dict3_list)):

            link_num = desired_dict3_list[i]["properties"]["id_object"]

            index = np.where(link_number == link_num)
            # print(index)
            # print(*index[0])

            if index[0].size > 0:
                # print(*index[0])
                if isinstance(location["properties"]["name"], str) and "Candidate MCS" in location["properties"]["name"]:           
                    if MCS_count not in MCS_ignored:
                        if index[0] == 266:
                            print('266')
                            print(target_id1)
                        if index[0] == 967:
                            print('967')
                            print(target_id1)
                        MCS_loc[index[0]] = 1
                        max_EV_MCS[index[0]] = config.max_EV_MCS[MCS_count]

                    MCS_check = 1
                else:
                    if FCS_count not in FCS_ignored:
                        FCS_loc[index[0]] = 1
                        FCS_flow_limit_vector[index[0]] = config.FCS_flow_limit_vector[FCS_count]
                        FCS_charged_dist[index[0]] = config.stoppage_time*config.FCS_range_per_hour[FCS_count]
                        max_EV_FCS[index[0]] = config.max_EV_FCS[FCS_count]



                    FCS_check = 1

                    

        # print(MCS_loc_group[MCS_count,*index[0]])
        MCS_count += MCS_check
        FCS_count += FCS_check


######### Path Planning ###########################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################

FCS_loc_indices = np.nonzero(FCS_loc)[0]
MCS_loc_indices = np.nonzero(MCS_loc)[0]
print(MCS_loc_indices)


def add_mcs_to_fcs(f_0, FCS_ports, num_MCS, EV_parking_ub, MCS_ports=config.MCS_flow_limit):
    # Step 1: how many MCS would be needed (ignoring availability)
    deficit = max(0, f_0 - FCS_ports)
    required = math.ceil(deficit / MCS_ports) if deficit > 0 else 0

    # Step 2: allocate what you can
    MCS_iter = min(required, num_MCS, EV_parking_ub)

    return MCS_iter



# def speed_calculate(start,end):

#     idx = basic_edges.index((start,end))
#     return link_speed[idx]

def heuristic_fun(start,end):
    # print(start)
    # print(end)
    return haversine(pos_nodes[int(start)][::-1],pos_nodes[int(end)][::-1],unit=Unit.KILOMETERS)  # returns shortest travel dist along Euclidean distance
 
G = nx.Graph()

G.add_edges_from(edge_information)

node_size = 1
node_color = 'black'

for i in pos_nodes.keys():
    if i in start_node:
        G.nodes[i]['color'] = node_color
        G.nodes[i]['size'] = node_size
        G.nodes[i]['ports'] = 0
        G.nodes[i]['charged_dist'] = 0
        G.nodes[i]['max_EV'] = 0
        G.nodes[i]['congestion'] = 0
        G.nodes[i]['type'] = 'node'
        G.nodes[i]['MCS_num'] = 0
        G.nodes[i]['full'] = 'no'

    elif i in goal_node:
        G.nodes[i]['color'] = node_color
        G.nodes[i]['size'] = node_size
        G.nodes[i]['ports'] = 0
        G.nodes[i]['charged_dist'] = 0
        G.nodes[i]['max_EV'] = 0
        G.nodes[i]['congestion'] = 0
        G.nodes[i]['type'] = 'node'
        G.nodes[i]['MCS_num'] = 0
        G.nodes[i]['full'] = 'no'

    else:
        G.nodes[i]['color'] = node_color
        G.nodes[i]['size'] = node_size
        G.nodes[i]['ports'] = 0
        G.nodes[i]['charged_dist'] = 0
        G.nodes[i]['max_EV'] = 0
        G.nodes[i]['congestion'] = 0
        G.nodes[i]['type'] = 'node'
        G.nodes[i]['MCS_num'] = 0
        G.nodes[i]['full'] = 'no'



count_FCS = 0
for i in FCS_loc_indices:
    count_FCS += 1
    FCS_node = num_nodes+count_FCS

    G.remove_edge(basic_edges[i][0],basic_edges[i][1])
    G.add_node(FCS_node,color=node_color,size=node_size,ports=FCS_flow_limit_vector[i],
               charged_dist=FCS_charged_dist[i],max_EV=max_EV_FCS[i],type='FCS',
               congestion=0,MCS_num=0,full='no')
    FCS_x = (0.5*pos_nodes[basic_edges[i][0]][0]) + (0.5*pos_nodes[basic_edges[i][1]][0])
    FCS_y = (0.5*pos_nodes[basic_edges[i][0]][1]) + (0.5*pos_nodes[basic_edges[i][1]][1])
    travel_time_temp = 0.5*road_travel_time[i]
    travel_dist_temp = 0.5*road_dist[i]
    G.add_edges_from([(basic_edges[i][0], FCS_node,{'travel_time':travel_time_temp,'road_limit':link_capacity[i],'travel_dist':travel_dist_temp}),\
                       (FCS_node, basic_edges[i][1],{'travel_time':travel_time_temp,'road_limit':link_capacity[i],'travel_dist':travel_dist_temp})])


    pos_nodes[FCS_node] = (FCS_x, FCS_y)
print(count_FCS)
count_MCS = 0
for i in MCS_loc_indices:
    count_MCS += 1
    MCS_node = num_nodes+count_FCS+count_MCS

    G.remove_edge(basic_edges[i][0],basic_edges[i][1])
    G.add_node(MCS_node,color=node_color,size=node_size,ports=0,
               charged_dist=MCS_charged_dist,max_EV=max_EV_MCS[i],type='MCS',
               congestion=0,MCS_num=0,full='no')
    MCS_x = (0.5*pos_nodes[basic_edges[i][0]][0]) + (0.5*pos_nodes[basic_edges[i][1]][0])
    MCS_y = (0.5*pos_nodes[basic_edges[i][0]][1]) + (0.5*pos_nodes[basic_edges[i][1]][1])
    travel_time_temp = 0.5*road_travel_time[i]
    travel_dist_temp = 0.5*road_dist[i]
    G.add_edges_from([(basic_edges[i][0], MCS_node,{'travel_time':travel_time_temp,'road_limit':link_capacity[i],'travel_dist':travel_dist_temp}),\
                       (MCS_node, basic_edges[i][1],{'travel_time':travel_time_temp,'road_limit':link_capacity[i],'travel_dist':travel_dist_temp})])


    pos_nodes[MCS_node] = (MCS_x, MCS_y)

OD_not_connected = []
OD_direct_route = []
OD_charged = []
OD_stranded = []
OD_timed_out = []
OD_paths_temp = {}
congestion_nume = np.zeros(count_FCS+count_MCS)
congestion_deno = np.zeros(count_FCS+count_MCS)

num_MCS = config.num_MCS

for i in range(config.num_od_pair):
    try:
        path = nx.astar_path(G,start_node[i],goal_node[i], heuristic=heuristic_fun, weight="travel_dist")
        

        if nx.astar_path_length(G,start_node[i],goal_node[i], heuristic=heuristic_fun, weight="travel_dist") <= config.init_range[i]:
           
           OD_paths_temp[i] = [int(x) for x in nx.astar_path(G,start_node[i],goal_node[i], heuristic=heuristic_fun, weight="travel_dist")]
           OD_direct_route.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i]))) 
        #    print('OD direct route',i)
        else:

            reachable_stations = []

            for j in range(count_FCS+count_MCS):
                if nx.astar_path_length(G,start_node[i],num_nodes+j+1, heuristic=heuristic_fun, weight="travel_dist") <= config.init_range[i]:

                    temp = nx.astar_path_length(G,start_node[i],num_nodes+j+1, heuristic=heuristic_fun, weight="travel_dist")
                    reachable_stations.append((num_nodes+j+1,float(temp)))

            
            if len(reachable_stations) == 0:
                OD_stranded.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i])))
                # print('OD stranded',i)
            else:
                
                if len(reachable_stations)>1:

                    stations_ordered = sorted(reachable_stations, key=lambda x: x[1])

                    node_number = 0
                    for ii in range(len(reachable_stations)):
                        if G.nodes[stations_ordered[ii][0]]['full'] == 'no':
                            node_number = stations_ordered[ii][0]
                            break

                    if node_number == 0:
                            node_number = stations_ordered[0][0]

                else:
                    node_number = stations_ordered[0][0]

                if config.init_flow[i] > G.nodes[node_number]['ports']:
                    MCS_iter = 0
                    MCS_iter = add_mcs_to_fcs(config.init_flow[i],G.nodes[node_number]['ports'],num_MCS,G.nodes[node_number]['max_EV'])

                    congestion_nume[node_number-num_nodes-1] += config.init_flow[i]
                    congestion_deno[node_number-num_nodes-1] += G.nodes[node_number]['ports'] + MCS_iter*config.MCS_flow_limit
                    G.nodes[node_number]['congestion'] = congestion_nume[node_number-num_nodes-1]/congestion_deno[node_number-num_nodes-1]
                    print('od-pair')
                    print(i)
                    print(node_number-num_nodes)

                    # G.nodes[node_number]['congestion'] += (config.init_flow[i]/(G.nodes[node_number]['ports'] + MCS_iter*config.MCS_flow_limit))
                    G.nodes[node_number]['MCS_num'] += int(MCS_iter)
                    num_MCS -= MCS_iter
                    G.nodes[node_number]['max_EV'] = G.nodes[node_number]['max_EV'] - MCS_iter
                    G.nodes[node_number]['ports'] = 0

                    # if G.nodes[node_number]['congestion'] > 1:
                    #     G.nodes[node_number]['full'] = 'yes'
                    # if G.nodes[node_number]['max_EV'] == 0:
                    #     G.nodes[node_number]['full'] = 'yes'



                    if G.nodes[node_number]['type'] == 'FCS':

                        if config.init_range[i] + min(MCS_charged_dist,G.nodes[node_number]['charged_dist']) - \
                            (nx.astar_path_length(G,start_node[i],node_number, heuristic=heuristic_fun, weight="travel_dist") + \
                            nx.astar_path_length(G,node_number,goal_node[i], heuristic=heuristic_fun, weight="travel_dist")) >= 0:
                            
                            path1 = [int(x) for x in nx.astar_path(G,start_node[i],node_number, heuristic=heuristic_fun, weight="travel_dist")]
                            path2 = [int(x) for x in nx.astar_path(G,node_number,goal_node[i], heuristic=heuristic_fun, weight="travel_dist")]

                            OD_paths_temp[i] = path1 + path2[1:]
                            # G.nodes[node_number]['color'] = 'cyan'
                            OD_charged.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i])))
                            # print('OD routed',i)
                        else:
                            OD_timed_out.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i])))
                            # print('OD timed out',i)

                    if G.nodes[node_number]['type'] == 'MCS':

                        if config.init_range[i] + MCS_charged_dist - \
                            (nx.astar_path_length(G,start_node[i],node_number, heuristic=heuristic_fun, weight="travel_dist") + \
                            nx.astar_path_length(G,node_number,goal_node[i], heuristic=heuristic_fun, weight="travel_dist")) >= 0:
                            
                            path1 = [int(x) for x in nx.astar_path(G,start_node[i],node_number, heuristic=heuristic_fun, weight="travel_dist")]
                            path2 = [int(x) for x in nx.astar_path(G,node_number,goal_node[i], heuristic=heuristic_fun, weight="travel_dist")]

                            OD_paths_temp[i] = path1 + path2[1:]
                            # G.nodes[node_number]['color'] = 'cyan'
                            OD_charged.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i])))
                            # print('OD routed',i)
                        else:
                            OD_timed_out.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i])))
                            # print('OD timed out',i)
                    
             
                else:


                    if G.nodes[node_number]['type'] == 'FCS':

                        G.nodes[node_number]['ports'] = G.nodes[node_number]['ports'] -  config.init_flow[i]

                        if config.init_range[i] + G.nodes[node_number]['charged_dist'] - \
                            (nx.astar_path_length(G,start_node[i],node_number, heuristic=heuristic_fun, weight="travel_dist") + \
                            nx.astar_path_length(G,node_number,goal_node[i], heuristic=heuristic_fun, weight="travel_dist")) >= 0:
                            
                            path1 = [int(x) for x in nx.astar_path(G,start_node[i],node_number, heuristic=heuristic_fun, weight="travel_dist")]
                            path2 = [int(x) for x in nx.astar_path(G,node_number,goal_node[i], heuristic=heuristic_fun, weight="travel_dist")]

                            OD_paths_temp[i] = path1 + path2[1:]
                            # G.nodes[node_number]['color'] = 'cyan'
                            OD_charged.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i])))
                            # print('OD routed',i)
                        else:
                            OD_timed_out.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i])))
                            # print('OD timed out',i)

                    if G.nodes[node_number]['type'] == 'MCS':

                        if config.init_range[i] + MCS_charged_dist - \
                            (nx.astar_path_length(G,start_node[i],node_number, heuristic=heuristic_fun, weight="travel_dist") + \
                            nx.astar_path_length(G,node_number,goal_node[i], heuristic=heuristic_fun, weight="travel_dist")) >= 0:
                            
                            path1 = [int(x) for x in nx.astar_path(G,start_node[i],node_number, heuristic=heuristic_fun, weight="travel_dist")]
                            path2 = [int(x) for x in nx.astar_path(G,node_number,goal_node[i], heuristic=heuristic_fun, weight="travel_dist")]

                            OD_paths_temp[i] = path1 + path2[1:]
                            # G.nodes[node_number]['color'] = 'cyan'
                            OD_charged.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i])))
                            # print('OD routed',i)
                        else:
                            OD_timed_out.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i])))
                            # print('OD timed out',i)
                


    except nx.NetworkXNoPath:
        OD_not_connected.append((int(config.start_node_centroid[i]),int(config.goal_node_centroid[i])))
        # print('OD not connected',i)



################################# Removing extra nodes and edges ##############################################
###############################################################################################################

# OD_paths = {}
# for i in range(config.num_od_pair):

#     OD_paths[i] = [x for x in OD_paths_temp[i] if x > 1000]


OD_paths = OD_paths_temp

True_MCS_loc = []
count_MCS_temp = 0
for i in MCS_loc_indices:
    count_MCS_temp += 1
    MCS_node_temp = num_nodes+count_FCS+count_MCS_temp

    if G.nodes[MCS_node_temp]['MCS_num'] == 0:
        MCS_loc[i] = 0
        G.remove_edge(basic_edges[i][0], MCS_node_temp)
        G.remove_edge(MCS_node_temp, basic_edges[i][1])
        G.remove_node(MCS_node_temp)
        G.add_edge(basic_edges[i][0],basic_edges[i][1],travel_time=road_travel_time[i],
                   road_limit=link_capacity[i],travel_dist=road_dist[i])
        del pos_nodes[MCS_node_temp]
    else:
        True_MCS_loc.append(MCS_node_temp)
###############################################################################################################
###############################################################################################################

print('Number of od pairs not connected:',len(OD_not_connected))
print('Number of od pairs directly going to target:',len(OD_direct_route))
print('Number of od pairs stopping at charging station:',len(OD_charged))
print('Number of od pairs stranded:',len(OD_stranded))
print('Number of od pairs timed out at FCS/MCS:',len(OD_timed_out))
print('Number of MCS left:',num_MCS)
# for node, attrs in G.nodes(data=True):
#     print(f"Node {node}")
#     for key, value in attrs.items():
#         print(f"  {key}: {value}")
# print(OD_paths)


######### Plotting ################################################################################################
###################################################################################################################


####################  Plotting the Normalised vehicle flow for edges ##############################################
###################################################################################################################

# nx.set_edge_attributes(G, 0, name='flow')    # initialize flow of each edge of the graph  to zero
# for key,value in OD_paths.items():
#     print(key)
#     print(value)
#     len_value = len(value)-1
#     for i in range(len_value):

#         G[value[i]][value[i+1]]['flow'] += (config.init_flow[key]/ G[value[i]][value[i+1]]['road_limit'])

# for u, v in G.edges:
#         w = G[u][v]['flow']

#         if w>= 0 and w<0.2:
#             G[u][v]['thickness'] = 1.5
#             G[u][v]['color'] = 'lightgreen'
#         if w>=0.2 and w<0.4:
#             G[u][v]['thickness'] = 3
#             G[u][v]['color'] = 'lawngreen'
#         if w>=0.4 and w<0.6:
#             G[u][v]['thickness'] = 4.5
#             G[u][v]['color'] = 'greenyellow'
#         if w>=0.6 and w<0.8:
#             G[u][v]['thickness'] = 6
#             G[u][v]['color'] = 'yellowgreen'
#         if w>=0.8 and w<=1:
#             G[u][v]['thickness'] = 7.5
#             G[u][v]['color'] = 'yellow'
#         if w>1 and w<1.2:
#             G[u][v]['thickness'] = 9
#             G[u][v]['color'] = 'orange'
#         if w>=1.2 and w<1.4:
#             G[u][v]['thickness'] = 10.5
#             G[u][v]['color'] = 'orangered'
#         if w>=1.4:
#             G[u][v]['thickness'] = 12
#             G[u][v]['color'] = 'red'

# color_range = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,2]
# color_names = ['lightgreen','lawngreen','greenyellow','yellowgreen','yellow','orange','orangered','red']

# cmap = colors.ListedColormap(color_names)
# norm = colors.BoundaryNorm(color_range, cmap.N)

# edge_thicknesses = [G[u][v]['thickness'] for u, v in G.edges()]
# edge_colors = [G[u][v]['color'] for u, v in G.edges()]
# node_colors = [data.get('color') for _, data in G.nodes(data=True)]
# node_sizes = [data.get('size') for _, data in G.nodes(data=True)]

# node_labels = {list(pos_nodes.keys())[i]:list(pos_nodes.keys())[i] for i in range(num_nodes)}   # to label only basic nodes

# fig, axs = plt.subplots()

# plt.margins(0.1)

# nx.draw(G,node_color=node_colors,node_size=node_sizes,pos=pos_nodes,edge_color=edge_colors,width=edge_thicknesses,ax=axs,
#         with_labels=False,labels=node_labels,font_color="purple",font_size=10)
# # Extract edge labels
# edge_labels = nx.get_edge_attributes(G, 'road_limit')
#  # Add colorbar legend
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # Only needed for matplotlib < 3.1
# cbar = plt.colorbar(sm, ax=axs, orientation='vertical')
# cbar.set_label('Normalised Flow values', rotation=270, labelpad=30,fontsize=28) # labelpad sets the gap between label ticks and the colorlegend title
# cbar.ax.tick_params(labelsize=20)  # Change tick label font size


####################  Plotting the Normalised vehicle flow for charging nodes #####################################
###################################################################################################################



for i in True_MCS_loc:

    w = G.nodes[i]['congestion']

    if w>= 0 and w<0.2:
        G.nodes[i]['size'] = 500
        G.nodes[i]['color'] = 'lightgreen'
    if w>=0.2 and w<0.4:
        G.nodes[i]['size'] = 500
        G.nodes[i]['color'] = 'yellowgreen'
    if w>=0.4 and w<0.6:
        G.nodes[i]['size'] = 500
        G.nodes[i]['color'] = 'darkkhaki'
    if w>=0.6 and w<0.8:
        G.nodes[i]['size'] = 500
        G.nodes[i]['color'] = 'khaki'
    if w>=0.8 and w<=1:
        G.nodes[i]['size'] = 500
        G.nodes[i]['color'] = 'navajowhite'
    if w>1 and w<1.2:
        G.nodes[i]['size'] = 500
        G.nodes[i]['color'] = 'lightsalmon'
    if w>=1.2 and w<1.4:
        G.nodes[i]['size'] = 500
        G.nodes[i]['color'] = 'salmon'
    if w>=1.4:
        G.nodes[i]['size'] = 500
        G.nodes[i]['color'] = 'coral'

for u, v in G.edges:
    G[u][v]['thickness'] = 1
    G[u][v]['color'] = 'black'

color_range = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,2]
color_names = ['lightgreen','yellowgreen','darkkhaki','khaki','navajowhite','lightsalmon','salmon','coral']

cmap = colors.ListedColormap(color_names)
norm = colors.BoundaryNorm(color_range, cmap.N)

edge_thicknesses = [G[u][v]['thickness'] for u, v in G.edges()]
edge_colors = [G[u][v]['color'] for u, v in G.edges()]
node_colors = [data.get('color') for _, data in G.nodes(data=True)]
node_sizes = [data.get('size') for _, data in G.nodes(data=True)]

# node_labels = {list(pos_nodes.keys())[i]:list(pos_nodes.keys())[i] for i in range(num_nodes)}   # to label only basic nodes

node_labels = {
        node: G.nodes[node]['MCS_num']
        for node, attrs in G.nodes(data=True)
        if attrs.get('MCS_num') > 0
    }

fig, axs = plt.subplots(dpi=100)
nx.draw(G,node_color=node_colors,node_size=node_sizes,pos=pos_nodes,edge_color=edge_colors,width=edge_thicknesses,ax=axs,
        with_labels=True,labels=node_labels,font_color="purple",font_size=30)
# Extract edge labels
edge_labels = nx.get_edge_attributes(G, 'road_limit')
plt.margins(0.1)
 # Add colorbar legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for matplotlib < 3.1
cbar = plt.colorbar(sm, ax=axs, orientation='vertical')
cbar.set_label('MCS utilization ratio', rotation=270, labelpad=30,fontsize=28) # labelpad sets the gap between label ticks and the colorlegend title
cbar.ax.tick_params(labelsize=20)  # Change tick label font size

plt.show()
