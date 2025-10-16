import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve(Network_matrix,od_pair_matrix,num_links,num_od_pair,
          road_travel_time,FCS_charging_time,MCS_charging_time,
          road_dist,FCS_charged_dist,MCS_charged_dist,init_range,
          road_flow_limit_vector,FCS_flow_limit_vector,
          init_flow,num_MCS,FCS_loc,MCS_loc,num_nodes,basic_edges,start_node,
           multiple_MCS_number,MCS_flow_limit,
           num_FCS_location,num_MCS_location,FCS_loc_group,MCS_loc_group,max_EV_FCS,max_EV_MCS):

    emerg_evac = gp.Model("multiple_od_pair")

    s_road = emerg_evac.addMVar((num_links,num_od_pair),vtype=GRB.BINARY,name="road_switch")
    s_FCS = emerg_evac.addMVar((num_links,num_od_pair),vtype=GRB.BINARY,name="FCS_switch")
    s_MCS = emerg_evac.addMVar((num_links,num_od_pair),vtype=GRB.BINARY,name="MCS_switch")
    # s_deployed = emerg_evac.addMVar((num_links,1),vtype=GRB.BINARY,name="MCS_deployed")
    # tmax = emerg_evac.addVar(lb=0.0,vtype=GRB.CONTINUOUS,name="max_time")
    tmax = emerg_evac.addMVar((1,num_od_pair),lb=0.0,vtype=GRB.CONTINUOUS,name="max_time")
    s_nodes = emerg_evac.addMVar((num_nodes,num_od_pair),lb=0.0,vtype=GRB.CONTINUOUS,name="network_node")
    MCS_limit = emerg_evac.addMVar((1,num_MCS_location),lb=0,ub=multiple_MCS_number,vtype=GRB.INTEGER,name="MCS_number")
    FCS_MCS_limit = emerg_evac.addMVar((1,num_FCS_location),lb=0,ub=multiple_MCS_number,vtype=GRB.INTEGER,name="FCS_with_MCS_number")

    FCS_loc_matrix = np.diag(FCS_loc)
    MCS_loc_matrix = np.diag(MCS_loc)


    Ones_vector = np.ones(num_links)

    basic_edges_matrix_temp = np.array(basic_edges)
    unique_vals = np.unique(basic_edges_matrix_temp)
    labels = np.arange(num_nodes)
    value_map = dict(zip(unique_vals, labels[:len(unique_vals)]))
    basic_edges_matrix = np.vectorize(value_map.get)(basic_edges_matrix_temp)

    road_dist_matrix = np.stack([road_dist]*num_od_pair, axis=1)
    MCS_charged_dist_matrix = np.stack([MCS_charged_dist.flatten()]*num_od_pair, axis=1)
    FCS_charged_dist_matrix = np.stack([FCS_charged_dist.flatten()]*num_od_pair, axis=1)

    for i in range(num_od_pair):
        if num_od_pair == 1:
            emerg_evac.addConstr(s_nodes[value_map[start_node[i]],0] == init_range)
        else:
            # emerg_evac.addConstr(s_nodes[value_map[start_node[i]],i] == init_range[i])
            emerg_evac.addConstr(s_nodes[value_map[start_node[i]],i] == init_range[0,i])

    if num_od_pair == 1:
        emerg_evac.addConstr((s_nodes[basic_edges_matrix[:,0],0] - road_dist - s_nodes[basic_edges_matrix[:,1],0]) * s_road[:,0] == 0)
        emerg_evac.addConstr((s_nodes[basic_edges_matrix[:,0],0] - road_dist + MCS_charged_dist - s_nodes[basic_edges_matrix[:,1],0]) * s_MCS[:,0] == 0)
        emerg_evac.addConstr((s_nodes[basic_edges_matrix[:,0],0] - road_dist + FCS_charged_dist - s_nodes[basic_edges_matrix[:,1],0]) * s_FCS[:,0] == 0)

    else:
         emerg_evac.addConstr((s_nodes[basic_edges_matrix[:,0],:] - road_dist_matrix - s_nodes[basic_edges_matrix[:,1],:]) * s_road == 0)
         emerg_evac.addConstr((s_nodes[basic_edges_matrix[:,0],:] - road_dist_matrix + MCS_charged_dist_matrix - s_nodes[basic_edges_matrix[:,1],:]) * s_MCS == 0)
         emerg_evac.addConstr((s_nodes[basic_edges_matrix[:,0],:] - road_dist_matrix + FCS_charged_dist_matrix - s_nodes[basic_edges_matrix[:,1],:]) * s_FCS == 0)


    od_pair_matrix = np.transpose(od_pair_matrix)
    
    if num_od_pair == 1:
        emerg_evac.addConstr((Network_matrix @ s_road[:,0])  +\
                              ((Network_matrix @ MCS_loc_matrix) @ s_MCS[:,0])  +\
                                  ((Network_matrix @ FCS_loc_matrix) @ s_FCS[:,0]) == od_pair_matrix)
    else:
        emerg_evac.addConstr((Network_matrix @ s_road) +\
                              ((Network_matrix @ MCS_loc_matrix) @ s_MCS)+\
                                  ((Network_matrix @ FCS_loc_matrix) @ s_FCS) == od_pair_matrix)     
    


    emerg_evac.addConstr((road_travel_time @ s_road) +\
                        (((FCS_charging_time + road_travel_time) @ FCS_loc_matrix) @ s_FCS)+\
                        (((MCS_charging_time + road_travel_time) @ MCS_loc_matrix) @ s_MCS) +\
                         - tmax == 0)

    emerg_evac.addConstr((road_dist @ s_road) -\
                        (((FCS_charged_dist - road_dist) @ FCS_loc_matrix) @ s_FCS) -\
                        (((MCS_charged_dist - road_dist) @ MCS_loc_matrix) @ s_MCS) <= init_range)



    
    if num_od_pair == 1:

        s_FCS_temp = FCS_loc_group * init_flow[0]
        s_MCS_temp = MCS_loc_group * init_flow[0]

        emerg_evac.addConstr((s_road[:,0] * init_flow[0]) - road_flow_limit_vector <= 0)
        emerg_evac.addConstr((s_FCS_temp) @ s_FCS[:,0] - FCS_flow_limit_vector - (MCS_flow_limit * FCS_MCS_limit[0,:])<= 0)
        # emerg_evac.addConstr((s_FCS_temp) @ s_FCS[:,0] - FCS_flow_limit_vector <= 0)
        emerg_evac.addConstr((s_MCS_temp) @ s_MCS[:,0] - (MCS_flow_limit * MCS_limit[0,:]) <= 0)

        emerg_evac.addConstr(FCS_flow_limit_vector + (MCS_flow_limit * FCS_MCS_limit[0,:]) <= max_EV_FCS)
        emerg_evac.addConstr(MCS_flow_limit * MCS_limit[0,:] <= max_EV_MCS)
    else:

        s_FCS_temp = FCS_loc_group @ s_FCS 
        s_MCS_temp = MCS_loc_group @ s_MCS

        emerg_evac.addConstr(s_road @ init_flow- road_flow_limit_vector <= 0)
        emerg_evac.addConstr((s_FCS_temp) @ init_flow - FCS_flow_limit_vector - (MCS_flow_limit * FCS_MCS_limit) <= 0)
        # emerg_evac.addConstr((s_FCS_temp) @ init_flow - FCS_flow_limit_vector <= 0)
        emerg_evac.addConstr((s_MCS_temp) @ init_flow - (MCS_flow_limit * MCS_limit) <= 0)

        emerg_evac.addConstr(FCS_flow_limit_vector + (MCS_flow_limit * FCS_MCS_limit) <= max_EV_FCS)
        emerg_evac.addConstr(MCS_flow_limit * MCS_limit <= max_EV_MCS)


    # emerg_evac.addConstr((MCS_loc_matrix @ s_MCS) - s_deployed <= 0)


    emerg_evac.addConstr((MCS_limit @ np.ones((num_MCS_location,1))) + (FCS_MCS_limit @ np.ones((num_FCS_location,1))) <= num_MCS)
    # emerg_evac.addConstr((MCS_limit) @ np.ones((num_links,1)) <= num_MCS)


    emerg_evac.addConstr(s_road + (FCS_loc_matrix @ s_FCS) <= 1)
    emerg_evac.addConstr(s_road + (MCS_loc_matrix @ s_MCS) <= 1)


    emerg_evac.addConstr((MCS_loc_matrix @ s_MCS) + (FCS_loc_matrix @ s_FCS) <= 1)


    # emerg_evac.addConstr(s_deployed + (FCS_loc_matrix @ s_FCS) <= 1)

    # obj_term = 0.2*(((Ones_vector @ s_road) +\
    #                   (Ones_vector @ s_FCS) +\
    #                   (Ones_vector @ s_MCS)) @ np.ones([num_od_pair,1]) +\
    #                      (MCS_limit @ np.ones((num_links,1))) )
    
    obj_term = 0.2*(((Ones_vector @ s_road) +\
                      (Ones_vector @ s_FCS) +\
                      (Ones_vector @ s_MCS)) @ np.ones([num_od_pair,1]) +\
                         ((MCS_limit @ np.ones((num_MCS_location,1))) + (FCS_MCS_limit @ np.ones((num_FCS_location,1)))) )
    
    obj_term1 = 1*(((Ones_vector @ s_road) +\
                      (Ones_vector @ s_FCS) +\
                      (Ones_vector @ s_MCS)) @ np.ones([num_od_pair,1]) +\
                         ((MCS_limit @ np.ones((num_MCS_location,1))) + (FCS_MCS_limit @ np.ones((num_FCS_location,1)))))

    emerg_evac.setObjective((1/num_od_pair)*(tmax @ np.ones([num_od_pair,1])) + obj_term, GRB.MINIMIZE)
    # emerg_evac.setObjective(obj_term1, GRB.MINIMIZE)

    emerg_evac.setParam("TimeLimit", 300)
    # emerg_evac.setParam("MIPGap", 21)

    emerg_evac.optimize()

    return emerg_evac