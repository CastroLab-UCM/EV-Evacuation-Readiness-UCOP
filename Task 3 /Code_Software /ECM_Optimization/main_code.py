import numpy as np
import matplotlib.pyplot as plt

import MILP_solve_t_avg_unidirectional # type: ignore
import plot_map_networkx_unidirectional # type: ignore
import json 
import pandas as pd # type: ignore
from haversine import haversine, Unit # type: ignore


## extract background traffic

df = pd.read_csv('LinkFlowStatistics_NormalFlow_v20250916a.txt')

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



with open('nodes.geojson', 'r') as f:
    nodes_data = json.load(f)


with open('sections.geojson', 'r') as f:
    edges_data = json.load(f)

with open('centroids.geojson', 'r') as f:
    centroids = json.load(f)

with open('centroid_connections.geojson', 'r') as f:
    centroid_connections = json.load(f)

with open('turnings.geojson', 'r') as f:
    turns = json.load(f)


######### Extracting OD pairs and initial flow ##################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################

start_node_centroid = np.array([51736,51731,50668,50662,50659,50677,50637,50640,50647,50653])
goal_node_centroid = np.array([50623,50626,50623,50626,50623,50626,50623,50626,50623,50626])


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

num_od_pair = 10  # change this parameter for multiple od pair
# init_range = 40*np.ones((1, num_od_pair))      # change this parameter for multiple od pair


init_flow = np.array([20,10,80,70,10,80,70,80,10,70])              # change this parameter for multiple od pair
init_range = np.array([[40,40,150,200,40,150,150,200,40,150]])     # change this parameter for multiple od pair
# init_flow = np.array([80,80,80,80])              # change this parameter for multiple od pair
# init_range = np.array([[40,40,40,40]])     # change this parameter for multiple od pair

# init_flow = np.array([20,10,80,10,80,80,70,10,50,20])              # change this parameter for multiple od pair
# init_range = np.array([[150,150,150,49,150,150,150,150,150,49]])     # change this parameter for multiple od pair

# init_flow = np.array([90,90])              # change this parameter for multiple od pair
# init_range = np.array([[35,40]])     # change this parameter for multiple od pair





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
        link_capacity_temp.append(location["properties"]["capacity"] - section_background_flow)
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


stoppage_time = 0.5
scaling_factor = 1/1
# FCS_range_per_hour = np.array([200,28,20,250,45,20,50,28,250])
# FCS_power_kW = np.array([50,7,6.5,250,16,6.5,16,7,150])
FCS_range_per_hour = np.array([200,0,0,200,0,200,0,0,2])
FCS_power_kW = np.array([150,0,0,150,0,150,0,0,150])
num_FCS_location = 9
FCS_charging_time = np.zeros(num_links)
FCS_charged_dist = np.zeros(num_links)

# FCS_flow_limit_vector = np.array([8,4,2,12,2,16,2,22,2])
FCS_flow_limit_vector = np.array([12,0,0,12,0,8,0,0,2])
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
            # print(*index[0])

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

            

multiple_MCS_number = 20

MCS_range_per_hour = 200

MCS_ports = 5
MCS_flow_limit = 5*scaling_factor

num_MCS = 20
MCS_flow_limit_vector = MCS_flow_limit*np.ones(num_MCS_location)

MCS_charging_time = stoppage_time*np.ones([1,num_links])
MCS_charged_dist = MCS_range_per_hour*MCS_charging_time
MCS_power_kW = 150

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


#################### saving output #############################################################################
################################################################################################################
################################################################################################################
################################################################################################################
# for v in solution.getVars():
#     if v.VarName.startswith("max_time"):
#         print('%s %g' % (v.VarName, v.X))

print("Optmization is done")


r_switch_values = [[solution.getVarByName(f"road_switch[{i},{j}]").X for j in range(num_od_pair)] for i in range(num_links)]
r_switch_values = np.array(r_switch_values)
FCS_switch_values = [[solution.getVarByName(f"FCS_switch[{i},{j}]").X for j in range(num_od_pair)] for i in range(num_links)]
FCS_switch_values = np.array(FCS_switch_values)
MCS_switch_values = [[solution.getVarByName(f"MCS_switch[{i},{j}]").X for j in range(num_od_pair)] for i in range(num_links)]
MCS_switch_values = np.array(MCS_switch_values)

# evac_time_op = [solution.getVarByName(f"max_time[{0},{j}]").X for j in range(num_od_pair)]
# evac_time_op = np.array(evac_time_op)

evac_time = []
for i in range(num_od_pair):

    total_time = np.dot(road_travel_time,r_switch_values[:,i])+\
                np.dot(road_travel_time,FCS_switch_values[:,i])+\
                np.dot(FCS_charging_time,FCS_switch_values[:,i])+\
                np.dot(road_travel_time,MCS_switch_values[:,i])+\
                np.dot(MCS_charging_time,MCS_switch_values[:,i])
    
    evac_time.append(total_time)
print(evac_time)
# print(evac_time_op)

number_of_MCS = [solution.getVarByName(f"MCS_number[{0},{j}]").X for j in range(num_MCS_location)]
number_of_MCS = np.array(number_of_MCS)
number_of_MCS_FCS = [solution.getVarByName(f"FCS_with_MCS_number[{0},{j}]").X for j in range(num_FCS_location)]
number_of_MCS_FCS = np.array(number_of_MCS_FCS)

MCS_number_scaled = MCS_loc_group * number_of_MCS[:, np.newaxis]
FCS_number_scaled = FCS_loc_group * number_of_MCS_FCS[:, np.newaxis]
FCS_flow_vector_scaled = FCS_loc_group * FCS_flow_limit_vector[:, np.newaxis]
MCS_flow_vector_scaled = MCS_loc_group * MCS_flow_limit_vector[:, np.newaxis]
FCS_power_scaled = FCS_loc_group * FCS_power_kW[:, np.newaxis]
FCS_range_scaled = FCS_loc_group * FCS_range_per_hour[:, np.newaxis]


MCS_number = MCS_number_scaled.sum(axis=0)
FCS_number = FCS_number_scaled.sum(axis=0)
FCS_flow_number = FCS_flow_vector_scaled.sum(axis=0)
MCS_flow_number = MCS_flow_vector_scaled.sum(axis=0)
FCS_power = FCS_power_scaled.sum(axis=0)
FCS_range = FCS_range_scaled.sum(axis=0)

sum_switches = r_switch_values + FCS_switch_values + MCS_switch_values
road_size = np.sum(sum_switches,axis=0)
basic_edges_numpy = np.array(basic_edges)



Mariposa_full_map = {}
Vehicles_dict = {}
chargers_dict = {}
charger_count = 0

for j in range(num_od_pair):

    temp_start_node = start_node[j]

    path = []
    subpath = []
    charging = {}
    path_list_temp = {}
    path_count = 0
    

    for i in range(int(road_size[j])):


        find_node1 = np.where(basic_edges_numpy[:, 0] == temp_start_node)[0]


        link_index1 = []



        if find_node1.size > 0:
            for k in range(find_node1.size):
                if r_switch_values[find_node1[k],j] + FCS_switch_values[find_node1[k],j] + MCS_switch_values[find_node1[k],j] == 1:
                    link_index1.append(find_node1[k])



        if len(link_index1) > 0:

            link_index = int(link_index1[0]) 

            if r_switch_values[link_index,j] == 1:

                path.append(int(link_number[link_index]))
                temp_start_node = basic_edges[link_index][1]

                if temp_start_node == goal_node[j]:
                    # print(path)
                    subpath.append(path.copy())
                    path.clear()
                    path_list_temp["subpath"+str(path_count+1)] = {'path':subpath[path_count],
                                                               'charging':{'cent_id':'N/A',
                                                                           'target_soc': -1,
                                                                           'charging time in hrs':-1,
                                                                           'battery usage in kWh':-1}}
                
                    break
            
            elif FCS_switch_values[link_index,j] == 1:


                path.append(int(link_number[link_index]))
                subpath.append(path.copy())
                path.clear()
                temp_start_node = basic_edges[link_index][1]

                desired_dict4 = next((feature for feature in centroid_connections["features"] if feature["properties"].get("id_object") == link_number[link_index]),None)

                cent_id = desired_dict4["properties"]["id_cent"]

                path_list_temp["subpath"+str(path_count+1)] = {'path': subpath[path_count],
                                                               'charging':{'cent_id': int(cent_id),
                                                                           'target_soc': 90,
                                                                           'charging time in hrs': FCS_charging_time[link_index],
                                                                           'battery usage in kWh': 0.2*FCS_range[link_index]*FCS_charging_time[link_index]}}
                path_count += 1

                chargers_dict['charging_station_'+str(charger_count+1)] = {"charging_station_id": int(cent_id),
                                                                           "fixed_charging_station": {"num_of_ports": int(FCS_flow_number[link_index]),
                                                                                                      "charger_power_in_kw": FCS_power[link_index]},
                                                                            "mobile_charging_station": {"battary_capacity_in_kwh": 50*MCS_flow_limit,
                                                                                                        "num_of_ports": MCS_ports*int(FCS_number[link_index]),
                                                                                                        "charger_power_in_kw": MCS_power_kW}                         
                                                                            }
                charger_count += 1
                # charging['charger_'+str(charger_count)] = {'cent_id':int(cent_id),
                #                                         'charging_time_in_hrs':FCS_charging_time[link_index],
                #                                         'charged_dist':int(FCS_charged_dist[link_index]),
                #                                         'Charger_ports':int(FCS_flow_number[link_index]),
                #                                         'number_of_chargers':int(FCS_number[link_index])}
                if temp_start_node == goal_node[j]:
                    break

            elif MCS_switch_values[link_index,j] == 1:

                path.append(int(link_number[link_index]))
                subpath.append(path.copy())
                path.clear()
                temp_start_node = basic_edges[link_index][1]

                desired_dict4 = next((feature for feature in centroid_connections["features"] if feature["properties"].get("id_object") == link_number[link_index]),None)

                cent_id = desired_dict4["properties"]["id_cent"]

                path_list_temp["subpath"+str(path_count+1)] = {'path':subpath[path_count],
                                                               'charging':{'cent_id':int(cent_id),
                                                                           'target_soc': 90,
                                                                           'charging time in hrs':MCS_charging_time[0,link_index],
                                                                           'battery usage in kWh':0.2*MCS_range_per_hour*MCS_charging_time[0,link_index]}}
                path_count += 1

                chargers_dict['charging_station_'+str(charger_count+1)] = {"charging_station_id": int(cent_id),
                                                                           "fixed_charging_station": {"num_of_ports": 0,
                                                                                                      "charger_power_in_kw": 0},
                                                                            "mobile_charging_station": {"battary_capacity_in_kwh": 50*MCS_flow_limit,
                                                                                                        "num_of_ports": MCS_ports*int(MCS_number[link_index]),
                                                                                                        "charger_power_in_kw": MCS_power_kW}                         
                                                                            }
                charger_count += 1
                

                # charging['charger_'+str(charger_count)] = {'cent_id':int(cent_id),
                #                                         'charging_time_in_hrs':MCS_charging_time[0,link_index],
                #                                         'charged_dist':int(MCS_charged_dist[0,link_index]),
                #                                         'Charger_ports':int(MCS_flow_number[link_index]),
                #                                         'number_of_chargers':int(MCS_number[link_index])}
                if temp_start_node == goal_node[j]:
                    break

            else:
                print('node not in network 1')

        else:
            print('node not in network 3')
        
        Vehicles_dict['vehicle_route_'+str(j+1)] = {'od_pair': (int(start_node_centroid[j]),int(goal_node_centroid[j])),
                                                    'init_flow': int(init_flow[j]),
                                                    'time_peroid': [0, 3600],
                                                    'initial_soc': 0.333*init_range[0,j],
                                                    'evac_time_hr': evac_time[j].item(),
                                                    'path_list': path_list_temp
                                                    }



    # Mariposa_full_map['vehicle_route_'+str(j+1)] = {'od_pair': (int(start_node_centroid[j]),int(goal_node_centroid[j])),
    #                                         'init_flow': int(init_flow[j]),
    #                                         'vehicles': [],
    #                                         'path': path,
    #                                         'charging': charging}
    
    Mariposa_full_map['vehicles'] = Vehicles_dict
    Mariposa_full_map['chargers'] = chargers_dict



with open('evacuation_result_Joseph2.json', 'w') as f:
    json.dump(Mariposa_full_map, f)


#################### Plotting ##################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

start = [[-120.4390,37.539],[-120.0241,37.137],[-120.3189,37.329],[-120.0083,37.417]]
goal = [[-119.6424,37.294],[-119.6424,37.294],[-119.6424,37.294],[-119.65512,37.3350]]

start = [[-119.9919,37.52],[-120.0427,37.456],[-119.9495,37.484],[-119.9602,37.496],[-119.9602,37.496],[-119.9602,37.496]]
goal = [[-120.4341,37.541],[-120.3060,37.274],[-119.7873,36.956],[-120.0348,36.961],[-120.0348,36.961],[-120.0348,36.961]]

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