import numpy as np

#################### Information about evacuees ################################################################

start_node_centroid = np.array([51736,51731,50668,50662,50659,50677,50637])
goal_node_centroid = np.array([50623,50626,50606,50611,50614,51851,51874])
# start_node_centroid = np.array([50637,50677,50659,50662,50668,51731,51736])
# goal_node_centroid = np.array([51874,51851,50614,50611,50606,50626,50623])
num_od_pair = len(start_node_centroid) 
init_flow = 60*np.ones(num_od_pair)      # unit: vehicle / hour
init_range = 20*np.ones(num_od_pair)  # unit: km

#################### Information about FCS and MCS #############################################################

stoppage_time = 7/60
scaling_factor = 1/stoppage_time

FCS_power_kW = np.array([150,0,0,250,0,250,0,0,120])
FCS_range_per_hour = (3.3*1.6)*FCS_power_kW                # EV efficiency = 3.3 miles/kWh = 3.3*1.6 km/kWh 
num_FCS_location = 9
FCS_flow_limit_vector = scaling_factor*np.array([12,0,0,12,0,8,0,0,2])
max_EV_FCS = np.array([70,25,50,50,5,70,30,70,20])

MCS_power_kW = 150 
MCS_range_per_hour = (3.3*1.6)*MCS_power_kW                # EV efficiency = 3.3 miles/kWh = 3.3*1.6 km/kWh                
num_MCS_location = 25
MCS_ports = 5
MCS_flow_limit = MCS_ports*scaling_factor
max_EV_MCS = np.array([5,5,10,5,20,10,30,20,20,20,5,5,20,20,10,10,5,5,20,20,10,5,10,5,10])
num_MCS = 25                   # Total number of MCS available in network
multiple_MCS_number = 100       # Maximum number of MCS in same location

#################### BPR function parameters ######################################################################

alpha = 0.15 # BPR function parameter
beta = 4 # BPR function parameter
