import numpy as np
import matplotlib.pyplot as plt

import MILP_solve_t_avg_unidirectional # type: ignore
import MILP_solve_tmax_unidirectional # type: ignore
import MILP_solve_t_avg_sd_unidirectional # type: ignore
import plot_map_networkx_unidirectional # type: ignore
import json 
import pandas as pd # type: ignore
from haversine import haversine, Unit # type: ignore
from pathlib import Path
import time

start_time = time.time()
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "GeoJson"

## extract background traffic

# df = pd.read_csv('C:\\Users\\josep\\Documents\\Codes\\IEEE journal\\Mariposa\\GeoJson\\LinkFlowStatistics_NormalFlow_v20250916a.txt')
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

############# Actual parameters #################################################################################
start_node_centroid = np.array([51736,51731,50668,50662,50659,50677,50637])
goal_node_centroid = np.array([50623,50626,50606,50611,50614,51851,51874])
num_od_pair = start_node_centroid.size  # change this parameter for multiple od pair
init_flow = 0.5*np.array([60,60,60,60,60,60,60])              # change this parameter for multiple od pair
init_range = 2*np.array([[10,10,10,10,10,10,10]])     # change this parameter for multiple od pair
#################################################################################################################


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

road_dist = np.zeros(num_links)

for i in range(num_links):
    road_dist[i] = haversine(pos_nodes[basic_edges[i][0]][::-1],pos_nodes[basic_edges[i][1]][::-1],unit=Unit.KILOMETERS)

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

                        

                    FCS_check = 1

                    FCS_charging_time[index[0]] = stoppage_time
                    FCS_charged_dist[index[0]] = stoppage_time*FCS_range_per_hour[FCS_count]

                    

        # print(MCS_loc_group[MCS_count,*index[0]])
        MCS_count += MCS_check
        FCS_count += FCS_check

            

multiple_MCS_number = 20      # Total number of MCS allowed per site

MCS_power_kW = 150
MCS_range_per_hour = (3.3*1.6)*MCS_power_kW

MCS_ports = 5
MCS_flow_limit = MCS_ports*scaling_factor

num_MCS = 25   # Total number of MCS available   
MCS_flow_limit_vector = MCS_flow_limit*np.ones(num_MCS_location)

MCS_charging_time = stoppage_time*np.ones(num_links)
MCS_charged_dist = MCS_range_per_hour*MCS_charging_time


# FCS_loc_trial = np.zeros(num_links)
# FCS_loc_group_trial = np.zeros((num_FCS_location,num_links),dtype=int)





#################### Optimization and plotting #################################################################
################################################################################################################
################################################################################################################
################################################################################################################
solve_parameter = "avg"



if solve_parameter == "worst":

    solution = MILP_solve_tmax_unidirectional.solve(Network_matrix,od_pair_matrix,num_links,num_od_pair,
                                road_travel_time,FCS_charging_time,MCS_charging_time,
                                road_dist,FCS_charged_dist,MCS_charged_dist,init_range,
                                link_capacity,FCS_flow_limit_vector,
                                init_flow,num_MCS,FCS_loc,MCS_loc,num_nodes,basic_edges,start_node,
                                multiple_MCS_number,MCS_flow_limit,
                                num_FCS_location,num_MCS_location,FCS_loc_group,MCS_loc_group,max_EV_FCS,max_EV_MCS)
    
if solve_parameter == "avg":

    solution = MILP_solve_t_avg_unidirectional.solve(Network_matrix,od_pair_matrix,num_links,num_od_pair,
                                road_travel_time,FCS_charging_time,MCS_charging_time,
                                road_dist,FCS_charged_dist,MCS_charged_dist,init_range,
                                link_capacity,FCS_flow_limit_vector,
                                init_flow,num_MCS,FCS_loc,MCS_loc,num_nodes,basic_edges,start_node,
                                multiple_MCS_number,MCS_flow_limit,
                                num_FCS_location,num_MCS_location,FCS_loc_group,MCS_loc_group,max_EV_FCS,max_EV_MCS)

if solve_parameter == "deviation":

    solution = MILP_solve_t_avg_sd_unidirectional.solve(Network_matrix,od_pair_matrix,num_links,num_od_pair,
                                road_travel_time,FCS_charging_time,MCS_charging_time,
                                road_dist,FCS_charged_dist,MCS_charged_dist,init_range,
                                link_capacity,FCS_flow_limit_vector,
                                init_flow,num_MCS,FCS_loc,MCS_loc,num_nodes,basic_edges,start_node,
                                multiple_MCS_number,MCS_flow_limit,
                                num_FCS_location,num_MCS_location,FCS_loc_group,MCS_loc_group,max_EV_FCS,max_EV_MCS)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Computation time: {elapsed_time:.6f} seconds")


    # np.save('solution.npy', solution, allow_pickle=True)


    # solution = np.load('solution.npy', allow_pickle=True).item()

# solution = MILP_solve_t_avg_v2.solve(Network_matrix,od_pair_matrix,num_links,num_od_pair,
#                  road_travel_time,FCS_charging_time,MCS_charging_time,
#                  road_dist,FCS_charged_dist,MCS_charged_dist,init_range,
#                  link_capacity,FCS_flow_limit_vector,
#                  init_flow,num_MCS,FCS_loc,MCS_loc,num_nodes,basic_edges,start_node,
#                  multiple_MCS_number,MCS_flow_limit)


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


r_switch_values = [[solution.getVarByName(f"road_switch[{i},{j}]").X for j in range(num_od_pair)] for i in range(num_links)]
r_switch_values = np.array(r_switch_values)
FCS_switch_values = [[solution.getVarByName(f"FCS_switch[{i},{j}]").X for j in range(num_od_pair)] for i in range(num_links)]
FCS_switch_values = np.array(FCS_switch_values)
MCS_switch_values = [[solution.getVarByName(f"MCS_switch[{i},{j}]").X for j in range(num_od_pair)] for i in range(num_links)]
MCS_switch_values = np.array(MCS_switch_values)

Switch_total = 60*(r_switch_values + FCS_switch_values + MCS_switch_values)

flow_normalised = Switch_total.sum(axis=1)/link_capacity

data = {'Normalised_flow':flow_normalised}
df = pd.DataFrame(data)

df.to_excel('Normalised_flow.xlsx', sheet_name='200', index=False, header=True)

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

data = {'Evac time':evac_time}
df = pd.DataFrame(data)


sum_switches = r_switch_values + FCS_switch_values + MCS_switch_values
road_size = np.sum(sum_switches,axis=0)
basic_edges_numpy = np.array(basic_edges)




#################### Plotting ##################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

# start = [[-120.4390,37.539],[-120.0241,37.137],[-120.3189,37.329],[-120.0083,37.417]]
# goal = [[-119.6424,37.294],[-119.6424,37.294],[-119.6424,37.294],[-119.65512,37.3350]]

# start = [[-119.9919,37.52],[-120.0427,37.456],[-119.9495,37.484],[-119.9602,37.496],[-119.9602,37.496],[-119.9602,37.496]]
# goal = [[-120.4341,37.541],[-120.3060,37.274],[-119.7873,36.956],[-120.0348,36.961],[-120.0348,36.961],[-120.0348,36.961]]

# start = [-119.9919,37.52]
# goal = [-120.4341,37.541]
        
# for i in range(num_od_pair):

#     if num_od_pair == 1:
#         plot_map_networkx_bidirectional.plot_mariposa(solution,0,num_od_pair,init_flow,FCS_loc,num_nodes,num_links,pos_nodes,basic_edges,start,goal)
#     else:
#         plot_map_networkx_bidirectional.plot_mariposa(solution,i,num_od_pair,init_flow[i],FCS_loc,num_nodes,num_links,pos_nodes,basic_edges,start[i],goal[i])

for i in range(num_od_pair):

    if num_od_pair == 1:
        plot_map_networkx_unidirectional.plot_mariposa(solution,0,num_od_pair,init_flow,FCS_loc,num_nodes,num_links,pos_nodes,basic_edges,FCS_loc_group,MCS_loc_group)
    else:
        plot_map_networkx_unidirectional.plot_mariposa(solution,i,num_od_pair,init_flow[i],FCS_loc,num_nodes,num_links,pos_nodes,basic_edges,FCS_loc_group,MCS_loc_group)




plt.show() 