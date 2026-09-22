import numpy as np
import matplotlib.pyplot as plt

import MILP_solve_t_avg_unidirectional # type: ignore
import json 
import matplotlib.colors as colors
import pandas as pd # type: ignore
from haversine import haversine, Unit # type: ignore
import networkx as nx # type: ignore
from pathlib import Path
from gurobipy import GRB

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "GeoJson"

## extract background traffic

df = pd.read_csv(DATA_DIR / 'LinkFlowStatistics_NormalFlow_v20250916a.txt')

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


######### Extracting OD pairs and initial flow ##################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################

start_node_centroid = np.array([51736,51731,50668,50662,50659,50677,50637])
goal_node_centroid = np.array([50623,50626,50606,50611,50614,51851,51874])

# start_node_centroid = np.array([51736,51731,50659,50653,50647,50677])
# goal_node_centroid = np.array([50623,50626,50623,50626,50623,50626])

# start_node_centroid = np.array([51736,51731,50659,50647])
# goal_node_centroid = np.array([50623,50626,50623,50623])

# start_node_centroid = np.array([50640,50640])
# goal_node_centroid = np.array([50626,50626])
start_node = []
goal_node = []

for start,goal in zip(start_node_centroid, goal_node_centroid):

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
# start_node =np.array([48524,48524])
# goal_node =np.array([49424,49424])

num_od_pair = 7  # change this parameter for multiple od pair
# init_range = 40*np.ones((1, num_od_pair))      # change this parameter for multiple od pair


init_flow = (60/80)*np.array([80,80,80,80,80,80,80])              # change this parameter for multiple od pair
# init_range = np.array([[40,40,150,200,40,150,150,200,40,150]])     # change this parameter for multiple od pair
# init_flow = np.array([80,80,80,80])              # change this parameter for multiple od pair
# init_range = np.array([[40,40,40,40]])     # change this parameter for multiple od pair

# init_flow = np.array([20,10,80,10,80,80,70,10,50,20])              # change this parameter for multiple od pair
# init_range = np.array([[150,150,150,49,150,150,150,150,150,49]])     # change this parameter for multiple od pair

# init_flow = np.array([10,10])              # change this parameter for multiple od pair
init_range = 2*np.array([[10,10,10,10,10,10,10]])     # change this parameter for multiple od pair





######### Extracting nodes and links ##############################################################################
###################################################################################################################
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

road_travel_time = np.zeros(num_links) 
road_dist = np.zeros(num_links)
edge_information = []

for i in range(num_links):
    road_dist[i] = haversine(pos_nodes[basic_edges[i][0]][::-1],pos_nodes[basic_edges[i][1]][::-1],unit=Unit.KILOMETERS)
    road_travel_time[i] = (1/link_speed[i])*road_dist[i]
    edge_information.append((basic_edges[i][0],basic_edges[i][1],{'travel_time':road_travel_time[i],
                                                                  'road_limit':link_capacity[i],
                                                                  'travel_dist':road_dist[i]}))

road_travel_time = (1/link_speed)*road_dist


Network_matrix = np.zeros((num_nodes,num_links))

row_count = 0
for key in pos_nodes.keys():
    col_count = 0
    for value in basic_edges:
        if value[0] == key:
            Network_matrix[row_count][col_count] = -1
            col_count += 1
        elif value[1] == key:
            Network_matrix[row_count][col_count] = 1
            col_count += 1
        else:
            col_count += 1
    row_count += 1

# np.save('Network_matrix.npy', Network_matrix)

od_pair_matrix = np.zeros((num_od_pair,num_nodes)) 

for i in range(num_od_pair):
    if num_od_pair == 1:
        od_pair_matrix[i][list(pos_nodes.keys()).index(start_node)] = -1
        od_pair_matrix[i][list(pos_nodes.keys()).index(goal_node)] = 1
    else:
        od_pair_matrix[i][list(pos_nodes.keys()).index(start_node[i])] = -1
        od_pair_matrix[i][list(pos_nodes.keys()).index(goal_node[i])] = 1


if num_od_pair == 1:
    od_pair_matrix = od_pair_matrix[0]







#################### Information about FCS and MCS #############################################################
################################################################################################################
################################################################################################################
################################################################################################################


stoppage_time = 7/60
scaling_factor = 1/stoppage_time
# FCS_range_per_hour = np.array([200,28,20,250,45,20,50,28,250])
# FCS_power_kW = np.array([50,7,6.5,250,16,6.5,16,7,150])
FCS_power_kW = np.array([150,0,0,250,0,250,0,0,120])
FCS_range_per_hour = (3.3*1.6)*FCS_power_kW                # EV efficiency = 3.3 miles/kWh = 3.3*1.6 km/kWh 

num_FCS_location = 9
FCS_charging_time = np.zeros(num_links)
FCS_charged_dist = np.zeros(num_links)

# FCS_flow_limit_vector = np.array([8,4,2,12,2,16,2,22,2])
FCS_flow_limit_vector = scaling_factor*np.array([12,0,0,12,0,8,0,0,2])
max_EV_FCS = scaling_factor*np.array([70,25,50,50,5,70,30,70,20])
max_EV_MCS = scaling_factor*np.array([5,5,10,5,20,10,30,20,20,20,5,5,20,20,10,10,5,5,20,20,10,5,10,5,10])

num_MCS_location = 25

FCS_loc = np.zeros(num_links)
MCS_loc = np.zeros(num_links)
FCS_loc_group = np.zeros((num_FCS_location,num_links),dtype=int)
MCS_loc_group = np.zeros((num_MCS_location,num_links),dtype=int)

FCS_loc_flow = np.zeros(num_links)

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
            # print(index[0])

            if index[0].size > 0:
                # print(*index[0])
                if isinstance(location["properties"]["name"], str) and "Candidate MCS" in location["properties"]["name"]:           
                    if MCS_count in MCS_ignored:
                        MCS_loc_group[MCS_count,*index[0]] = 0
                        MCS_loc[index[0]] = 0
                    else:
                        MCS_loc_group[MCS_count,*index[0]] = 1
                        MCS_loc[index[0]] = 1

                    MCS_check = 1
                else:
                    if FCS_count in FCS_ignored:
                        FCS_loc_group[FCS_count,*index[0]] = 0
                        FCS_loc[index[0]] = 0
                    else:
                        FCS_loc_group[FCS_count,*index[0]] = 1
                        FCS_loc[index[0]] = 1
                        FCS_loc_flow[index[0]] = FCS_flow_limit_vector[FCS_count]

                    FCS_check = 1

                    FCS_charging_time[index[0]] = stoppage_time
                    FCS_charged_dist[index[0]] = stoppage_time*FCS_range_per_hour[FCS_count]

        # print(MCS_loc_group[MCS_count,*index[0]])
        MCS_count += MCS_check
        FCS_count += FCS_check

FCS_loc_indices = np.nonzero(FCS_loc)[0]
MCS_loc_indices = np.nonzero(MCS_loc)[0]           

multiple_MCS_number = 20

MCS_power_kW = 150
MCS_range_per_hour = (3.3*1.6)*MCS_power_kW

MCS_ports = 5
MCS_flow_limit = MCS_ports*scaling_factor

num_MCS = 25
MCS_flow_limit_vector = MCS_flow_limit*np.ones(num_MCS_location)

MCS_charging_time = stoppage_time*np.ones(num_links)
MCS_charged_dist = MCS_range_per_hour*MCS_charging_time



FCS_loc_trial = np.zeros(num_links)
FCS_loc_group_trial = np.zeros((num_FCS_location,num_links),dtype=int)


#################### Optimization and plotting #################################################################
################################################################################################################
################################################################################################################
################################################################################################################


solution = MILP_solve_t_avg_unidirectional.solve(Network_matrix,od_pair_matrix,num_links,num_od_pair,
                            road_travel_time,FCS_charging_time,MCS_charging_time,
                            road_dist,FCS_charged_dist,MCS_charged_dist,init_range,
                            link_capacity,FCS_flow_limit_vector,
                            init_flow,num_MCS,FCS_loc,MCS_loc,num_nodes,basic_edges,start_node,
                            multiple_MCS_number,MCS_flow_limit,
                            num_FCS_location,num_MCS_location,FCS_loc_group,MCS_loc_group,max_EV_FCS,max_EV_MCS)
    




    # np.save('solution.npy', solution, allow_pickle=True)


    # solution = np.load('solution.npy', allow_pickle=True).item()

# solution = MILP_solve_t_avg_v2.solve(Network_matrix,od_pair_matrix,num_links,num_od_pair,
#                  road_travel_time,FCS_charging_time,MCS_charging_time,
#                  road_dist,FCS_charged_dist,MCS_charged_dist,init_range,
#                  link_capacity,FCS_flow_limit_vector,
#                  init_flow,num_MCS,FCS_loc,MCS_loc,num_nodes,basic_edges,start_node,
#                  multiple_MCS_number,MCS_flow_limit)


# ============================================================
# CHECK OPTIMIZATION STATUS
# ============================================================
# The rest of this script is executed ONLY when Gurobi has at
# least one feasible solution.
#
# If the time limit is reached:
#   - SolCount > 0  -> continue with the best feasible solution
#   - SolCount == 0 -> skip all remaining calculations/plots
# ============================================================

has_feasible_solution = solution.SolCount > 0

if solution.Status == GRB.OPTIMAL:

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY")
    print("Solution status: OPTIMAL")
    print(f"Number of feasible solutions found: {solution.SolCount}")
    print("="*60 + "\n")


elif solution.Status == GRB.TIME_LIMIT:

    print("\n" + "="*60)
    print("OPTIMIZATION TIME LIMIT EXPIRED")
    print(f"Number of feasible solutions found: {solution.SolCount}")

    if has_feasible_solution:
        print("A feasible solution was found before the time limit.")
        print("Optimality was not proven.")
        print("The program will continue using the best feasible solution found.")
    else:
        print("NO FEASIBLE SOLUTION WAS FOUND BEFORE THE TIME LIMIT.")
        print("The remaining calculations and plots will NOT be executed.")
        print("Consider increasing the Gurobi time limit in")
        print("MILP_solve_t_avg_unidirectional.py and running again.")

    print("="*60 + "\n")


elif solution.Status == GRB.INFEASIBLE:

    print("\n" + "="*60)
    print("OPTIMIZATION IS INFEASIBLE")
    print("No feasible solution exists for the current optimization model.")
    print("The remaining calculations and plots will NOT be executed.")
    print("="*60 + "\n")


elif has_feasible_solution:

    print("\n" + "="*60)
    print("OPTIMIZATION TERMINATED WITHOUT PROVING OPTIMALITY")
    print(f"Gurobi status code: {solution.Status}")
    print(f"Number of feasible solutions found: {solution.SolCount}")
    print("The program will continue using the best feasible solution found.")
    print("="*60 + "\n")


else:

    print("\n" + "="*60)
    print("NO FEASIBLE SOLUTION WAS FOUND")
    print(f"Gurobi status code: {solution.Status}")
    print(f"Number of feasible solutions found: {solution.SolCount}")
    print("The remaining calculations and plots will NOT be executed.")
    print("Consider increasing the Gurobi time limit in")
    print("MILP_solve_t_avg_unidirectional.py and running again.")
    print("="*60 + "\n")


# ============================================================
# RUN POST-PROCESSING AND PLOTTING ONLY IF A FEASIBLE SOLUTION EXISTS
# ============================================================

if has_feasible_solution:
    #################### saving output #############################################################################
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    # for v in solution.getVars():
    #     if v.VarName.startswith("max_time"):
    #         print('%s %g' % (v.VarName, v.X))

    print("Optimization is done")

    # for var in solution.getVars():
    #     if var.VarName.startswith('temp_var'):
    #         print('aux: ', var.X)
    #     if var.VarName.startswith('deltaT'):
    #         print('deltaT: ', var.X)
    FCS_switch = [var for var in solution.getVars() if var.VarName.startswith("FCS_switch")]
    MCS_switch = [var for var in solution.getVars() if var.VarName.startswith("MCS_switch")]


    r_switch_values = [[solution.getVarByName(f"road_switch[{i},{j}]").X for j in range(num_od_pair)] for i in range(num_links)]
    r_switch_values = np.array(r_switch_values)
    FCS_switch_values = [[solution.getVarByName(f"FCS_switch[{i},{j}]").X for j in range(num_od_pair)] for i in range(num_links)]
    FCS_switch_values = np.array(FCS_switch_values)
    MCS_switch_values = [[solution.getVarByName(f"MCS_switch[{i},{j}]").X for j in range(num_od_pair)] for i in range(num_links)]
    MCS_switch_values = np.array(MCS_switch_values)

    number_of_MCS = [solution.getVarByName(f"MCS_number[{0},{j}]").X for j in range(num_MCS_location)]
    number_of_MCS = np.array(number_of_MCS)
    number_of_MCS_FCS = [solution.getVarByName(f"FCS_with_MCS_number[{0},{j}]").X for j in range(num_FCS_location)]
    number_of_MCS_FCS = np.array(number_of_MCS_FCS)
    print('Total MCS used:',number_of_MCS.sum()+number_of_MCS_FCS.sum())
    MCS_number_scaled = MCS_loc_group * number_of_MCS[:, np.newaxis]
    MCS_number = MCS_number_scaled.sum(axis=0)


    FCS_number_scaled = FCS_loc_group * number_of_MCS_FCS[:, np.newaxis]
    FCS_number = FCS_number_scaled.sum(axis=0)

    Switch_total = 60*(r_switch_values + FCS_switch_values + MCS_switch_values)

    flow_normalised = Switch_total.sum(axis=1)/link_capacity
    flow_total = Switch_total.sum(axis=1)


    evac_time = []
    evac_dist = []
    for i in range(num_od_pair):

        total_time = np.dot(road_travel_time,r_switch_values[:,i])+\
                    np.dot(road_travel_time,FCS_switch_values[:,i])+\
                    np.dot(FCS_charging_time,FCS_switch_values[:,i])+\
                    np.dot(road_travel_time,MCS_switch_values[:,i])+\
                    np.dot(MCS_charging_time,MCS_switch_values[:,i])
    
        total_dist = np.dot(road_dist,r_switch_values[:,i])+\
                    np.dot(road_dist,FCS_switch_values[:,i])+\
                    np.dot(road_dist,MCS_switch_values[:,i])
    
        evac_time.append(total_time)
        evac_dist.append(total_dist)
    # print(evac_time)
    # print(evac_dist)



    ######### Plotting ################################################################################################
    ###################################################################################################################


    ####################  Plotting the Normalised vehicle flow for edges ##############################################
    ###################################################################################################################

    G = nx.Graph()

    G.add_edges_from(edge_information)

    for (edge, flow) in zip(basic_edges, flow_normalised):
        # print(edge[1])
        G.edges[edge]['flow'] = flow

    node_size = 1
    node_color = 'black'

    for i in pos_nodes.keys():
        if i in start_node:
            G.nodes[i]['color'] = node_color
            G.nodes[i]['size'] = node_size
            G.nodes[i]['MCS_num'] = 0
            G.nodes[i]['congestion'] = 0

        elif i in goal_node:
            G.nodes[i]['color'] = node_color
            G.nodes[i]['size'] = node_size
            G.nodes[i]['MCS_num'] = 0
            G.nodes[i]['congestion'] = 0

        else:
            G.nodes[i]['color'] = node_color
            G.nodes[i]['size'] = node_size
            G.nodes[i]['MCS_num'] = 0
            G.nodes[i]['congestion'] = 0


    FCS_indices = FCS_switch_values.sum(axis=1)
    MCS_indices = MCS_switch_values.sum(axis=1)
    r_indices = r_switch_values.sum(axis=1)

    count_FCS = 0


    for i in range(num_links):

        if FCS_indices[i] >= 1:
            count_FCS += 1
            FCS_node = num_nodes+count_FCS
            G.remove_edge(basic_edges[i][0],basic_edges[i][1])
            G.add_node(FCS_node,color=node_color,size=node_size,MCS_num=int(FCS_number[i]),congestion=((flow_total[i])/((FCS_loc_flow[i])+(int(FCS_number[i])*MCS_flow_limit*(FCS_indices[i]+MCS_indices[i]+r_indices[i])))))
            FCS_x = (0.5*pos_nodes[basic_edges[i][0]][0]) + (0.5*pos_nodes[basic_edges[i][1]][0])
            FCS_y = (0.5*pos_nodes[basic_edges[i][0]][1]) + (0.5*pos_nodes[basic_edges[i][1]][1])
            travel_time_temp = 0.5*road_travel_time[i]
            road_dist_temp = 0.5*road_dist[i]
            G.add_edges_from([(basic_edges[i][0], FCS_node,{'flow':flow_normalised[i],'travel_time':travel_time_temp,'road_limit':link_capacity[i],'travel_dist':road_dist_temp}),\
                            (FCS_node, basic_edges[i][1],{'flow':flow_normalised[i],'travel_time':travel_time_temp,'road_limit':link_capacity[i],'travel_dist':road_dist_temp})])


            pos_nodes[FCS_node] = (FCS_x, FCS_y)



    count_MCS1 = 0


    for i in range(num_links):

        if MCS_indices[i] >= 1:
            count_MCS1 += 1
            MCS_node1 = num_nodes+count_FCS+count_MCS1
            G.remove_edge(basic_edges[i][0],basic_edges[i][1])
            G.add_node(MCS_node1,color=node_color,size=node_size,MCS_num= int(MCS_number[i]),congestion=((flow_total[i])/(int(MCS_number[i])*MCS_flow_limit*(FCS_indices[i]+MCS_indices[i]+r_indices[i]))))
            MCS_x = (0.5*pos_nodes[basic_edges[i][0]][0]) + (0.5*pos_nodes[basic_edges[i][1]][0])
            MCS_y = (0.5*pos_nodes[basic_edges[i][0]][1]) + (0.5*pos_nodes[basic_edges[i][1]][1])
            travel_time_temp = 0.5*road_travel_time[i]
            road_dist_temp = 0.5*road_dist[i]
            G.add_edges_from([(basic_edges[i][0], MCS_node1,{'flow':flow_normalised[i],'travel_time':travel_time_temp,'road_limit':link_capacity[i],'travel_dist':road_dist_temp}),\
                            (MCS_node1, basic_edges[i][1],{'flow':flow_normalised[i],'travel_time':travel_time_temp,'road_limit':link_capacity[i],'travel_dist':road_dist_temp})])


            pos_nodes[MCS_node1] = (MCS_x, MCS_y)

    # count_MCS2 = 0

    # for jj in range(num_od_pair):
    #     for i in range(num_links):
    #         if MCS_switch_values[i,jj] == 1:
    #             count_MCS2 += 1
    #             MCS_node2 = num_nodes+count_FCS+count_MCS2
    #             G.nodes[MCS_node2]['congestion'] = ((init_flow[jj])/(int(MCS_number[i])*MCS_flow_limit))


    for u, v in G.edges:
            w = G[u][v]['flow']

            if w>= 0 and w<0.2:
                G[u][v]['thickness'] = 1.5
                G[u][v]['color'] = 'lightgreen'
            if w>=0.2 and w<0.4:
                G[u][v]['thickness'] = 3
                G[u][v]['color'] = 'lawngreen'
            if w>=0.4 and w<0.6:
                G[u][v]['thickness'] = 4.5
                G[u][v]['color'] = 'greenyellow'
            if w>=0.6 and w<0.8:
                G[u][v]['thickness'] = 6
                G[u][v]['color'] = 'yellowgreen'
            if w>=0.8 and w<=1:
                G[u][v]['thickness'] = 7.5
                G[u][v]['color'] = 'yellow'
            if w>1 and w<1.2:
                G[u][v]['thickness'] = 9
                G[u][v]['color'] = 'orange'
            if w>=1.2 and w<1.4:
                G[u][v]['thickness'] = 10.5
                G[u][v]['color'] = 'orangered'
            if w>=1.4:
                G[u][v]['thickness'] = 12
                G[u][v]['color'] = 'red'

    color_range = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,2]
    color_names = ['lightgreen','lawngreen','greenyellow','yellowgreen','yellow','orange','orangered','red']

    cmap = colors.ListedColormap(color_names)
    norm = colors.BoundaryNorm(color_range, cmap.N)

    edge_thicknesses = [G[u][v]['thickness'] for u, v in G.edges()]
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    node_colors = [data.get('color') for _, data in G.nodes(data=True)]
    node_sizes = [data.get('size') for _, data in G.nodes(data=True)]

    node_labels = {list(pos_nodes.keys())[i]:list(pos_nodes.keys())[i] for i in range(num_nodes)}   # to label only basic nodes

    fig, axs = plt.subplots(dpi=100)

    plt.margins(0.1)

    nx.draw(G,node_color=node_colors,node_size=node_sizes,pos=pos_nodes,edge_color=edge_colors,width=edge_thicknesses,ax=axs,
            with_labels=False,labels=node_labels,font_color="purple",font_size=10)
    # Extract edge labels
    edge_labels = nx.get_edge_attributes(G, 'road_limit')
     # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for matplotlib < 3.1
    cbar = plt.colorbar(sm, ax=axs, orientation='vertical')
    cbar.set_label('Normalised Flow values', rotation=270, labelpad=30,fontsize=28) # labelpad sets the gap between label ticks and the colorlegend title
    cbar.ax.tick_params(labelsize=20)  # Change tick label font size


    ####################  Plotting the Normalised vehicle flow for charging nodes #####################################
    ###################################################################################################################
    count_MCS_temp = 0
    for i in range(num_links):

        if MCS_indices[i] >= 1:
            count_MCS_temp += 1
            MCS_node_temp = num_nodes+count_FCS+count_MCS_temp
            w = G.nodes[MCS_node_temp]['congestion']

            if w>= 0 and w<0.2:
                G.nodes[MCS_node_temp]['size'] = 500
                G.nodes[MCS_node_temp]['color'] = 'lightgreen'
            if w>=0.2 and w<0.4:
                G.nodes[MCS_node_temp]['size'] = 500
                G.nodes[MCS_node_temp]['color'] = 'yellowgreen'
            if w>=0.4 and w<0.6:
                G.nodes[MCS_node_temp]['size'] = 500
                G.nodes[MCS_node_temp]['color'] = 'darkkhaki'
            if w>=0.6 and w<0.8:
                G.nodes[MCS_node_temp]['size'] = 500
                G.nodes[MCS_node_temp]['color'] = 'khaki'
            if w>=0.8 and w<=1:
                G.nodes[MCS_node_temp]['size'] = 500
                G.nodes[MCS_node_temp]['color'] = 'navajowhite'
            if w>1 and w<1.2:
                G.nodes[MCS_node_temp]['size'] = 500
                G.nodes[MCS_node_temp]['color'] = 'lightsalmon'
            if w>=1.2 and w<1.4:
                G.nodes[MCS_node_temp]['size'] = 500
                G.nodes[MCS_node_temp]['color'] = 'salmon'
            if w>=1.4:
                G.nodes[MCS_node_temp]['size'] = 500
                G.nodes[MCS_node_temp]['color'] = 'coral'

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
            with_labels=True,labels=node_labels,font_color="purple",font_size=20)
    # Extract edge labels
    edge_labels = nx.get_edge_attributes(G, 'road_limit')
    plt.margins(0.1)
     # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for matplotlib < 3.1
    cbar = plt.colorbar(sm, ax=axs, orientation='vertical')
    cbar.set_label('MCS utilization ratio', rotation=270, labelpad=30,fontsize=28) # labelpad sets the gap between label ticks and the colorlegend title
    cbar.ax.tick_params(labelsize=20)  # Change tick label font size
    # plt.savefig("my_plot.png", dpi=300, bbox_inches="tight")

    plt.show() 
