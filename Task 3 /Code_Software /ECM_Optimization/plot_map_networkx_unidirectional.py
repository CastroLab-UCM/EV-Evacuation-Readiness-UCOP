import numpy as np
import matplotlib.pyplot as plt
import networkx as nx # type: ignore


def plot_mariposa(model,od_pair_num,num_od_pair,init_flow,FCS_loc,num_nodes,num_links,pos_nodes,basic_edges,FCS_loc_group,MCS_loc_group):

    
    fig, axs = plt.subplots()
    road_switch = [var for var in model.getVars() if var.VarName.startswith("road_switch")]
    FCS_switch = [var for var in model.getVars() if var.VarName.startswith("FCS_switch")]
    MCS_switch = [var for var in model.getVars() if var.VarName.startswith("MCS_switch")]
    MCS_limit = [var.X for var in model.getVars() if var.VarName.startswith("MCS_number")]
    FCS_MCS_limit = [var.X for var in model.getVars() if var.VarName.startswith("FCS_with_MCS_number")]


    MCS_number_scaled = MCS_loc_group * np.array(MCS_limit)[:, np.newaxis]
    FCS_number_scaled = FCS_loc_group * np.array(FCS_MCS_limit)[:, np.newaxis]

    MCS_number = MCS_number_scaled.sum(axis=0)
    FCS_number = FCS_number_scaled.sum(axis=0)

    node_colors = []
    node_colors.extend(['grey']*num_nodes)

    node_sizes = []
    node_sizes.extend([25]*num_nodes)
    Node_size = 200

    edge_thickness = [1,2]
    edge_color = ['black','blue']

    FCS_color = ['red','green']
    MCS_color = ['yellow','orange']

    alpha = 0.4

    G = nx.DiGraph()

    G.add_edges_from([*basic_edges])
    G.add_nodes_from(G.nodes, MCS_count=0)

    # for node in G.nodes:
    #     G.nodes[node]["MCS_count"] = 0
    
    
    # To add FCS to each link
    count_FCS = 0

    for i in range(num_links):

        if FCS_loc[i] == 1:

            count_FCS += 1
            FCS_node = num_nodes+count_FCS

            G.remove_edge(basic_edges[i][0],basic_edges[i][1])
            G.add_node(FCS_node)
            G.nodes[FCS_node]['MCS_count'] = int(FCS_number[i])

            FCS_x = ((1-alpha)*pos_nodes[basic_edges[i][0]][0]) + (alpha*pos_nodes[basic_edges[i][1]][0])
            FCS_y = ((1-alpha)*pos_nodes[basic_edges[i][0]][1]) + (alpha*pos_nodes[basic_edges[i][1]][1])
            G.add_edges_from([(basic_edges[i][0], FCS_node), (FCS_node, basic_edges[i][1])])

            pos_nodes[FCS_node] = (FCS_x, FCS_y)

            node_sizes.append(Node_size)
            if FCS_switch[od_pair_num + i*num_od_pair].X == 1:
                node_colors.append(FCS_color[1])             
            else:
                node_colors.append(FCS_color[0])
       



        


    # To add MCS to each link
    count_MCS = 0 
    count = 0

    for i in range(num_links):
        if MCS_number[i] >= 1:   
            count_MCS += 1
            MCS_node = num_nodes+count_FCS+count_MCS
            G.add_node(MCS_node)
            G.nodes[MCS_node]['MCS_count'] = int(MCS_number[i])
            if FCS_loc[i] == 1:  
                count += 1
                temp_node_num1 = num_nodes+count
                temp_node_num2 = list(G.successors(temp_node_num1))[0]           
            else:
                temp_node_num1 = basic_edges[i][0]
                temp_node_num2 = basic_edges[i][1]

            MCS_x = ((1-alpha)*pos_nodes[temp_node_num1][0]) + (alpha*pos_nodes[temp_node_num2][0])
            MCS_y = ((1-alpha)*pos_nodes[temp_node_num1][1]) + (alpha*pos_nodes[temp_node_num2][1])
            G.remove_edge(temp_node_num1,temp_node_num2)
            G.add_edges_from([(temp_node_num1, MCS_node), (MCS_node, temp_node_num2)])
            pos_nodes[MCS_node] = (MCS_x, MCS_y)


            node_sizes.append(Node_size)
            if MCS_switch[od_pair_num + i*num_od_pair].X == 1:
                node_colors.append(MCS_color[1])
            else:
                node_colors.append(MCS_color[0])

            
    count1 = 0
    count2 = 0
    for i in range(num_links):
        if G.has_edge(basic_edges[i][0], basic_edges[i][1]) == True:

            if road_switch[od_pair_num + i*num_od_pair].X == 1:
                G[basic_edges[i][0]][basic_edges[i][1]]['color'] = edge_color[1]
                G[basic_edges[i][0]][basic_edges[i][1]]['thickness'] = edge_thickness[1]
            else:
                G[basic_edges[i][0]][basic_edges[i][1]]['color'] = edge_color[0]
                G[basic_edges[i][0]][basic_edges[i][1]]['thickness'] = edge_thickness[0]

        else:
            if FCS_loc[i] == 1 and MCS_number[i] == 0:
                count1 += 1
                temp1 = list(G.out_edges(num_nodes+count1))
                temp2 = list(G.in_edges(num_nodes+count1))

                if FCS_switch[od_pair_num + i*num_od_pair].X == 1 or road_switch[od_pair_num + i*num_od_pair].X == 1:
                    G[temp1[0][0]][temp1[0][1]]['color'] = edge_color[1]
                    G[temp1[0][0]][temp1[0][1]]['thickness'] = edge_thickness[1]
                    G[temp2[0][0]][temp2[0][1]]['color'] = edge_color[1]
                    G[temp2[0][0]][temp2[0][1]]['thickness'] = edge_thickness[1]
                else:
                    G[temp1[0][0]][temp1[0][1]]['color'] = edge_color[0]
                    G[temp1[0][0]][temp1[0][1]]['thickness'] = edge_thickness[0]
                    G[temp2[0][0]][temp2[0][1]]['color'] = edge_color[0]
                    G[temp2[0][0]][temp2[0][1]]['thickness'] = edge_thickness[0]

            if FCS_loc[i] == 0 and MCS_number[i] >= 1:
                count2 += 1
                temp3 = list(G.out_edges(num_nodes+count_FCS+count2))
                temp4 = list(G.in_edges(num_nodes+count_FCS+count2))

                if MCS_switch[od_pair_num + i*num_od_pair].X == 1 or road_switch[od_pair_num + i*num_od_pair].X == 1:
                    G[temp3[0][0]][temp3[0][1]]['color'] = edge_color[1]
                    G[temp3[0][0]][temp3[0][1]]['thickness'] = edge_thickness[1]
                    G[temp4[0][0]][temp4[0][1]]['color'] = edge_color[1]
                    G[temp4[0][0]][temp4[0][1]]['thickness'] = edge_thickness[1]
                else:
                    G[temp3[0][0]][temp3[0][1]]['color'] = edge_color[0]
                    G[temp3[0][0]][temp3[0][1]]['thickness'] = edge_thickness[0]
                    G[temp4[0][0]][temp4[0][1]]['color'] = edge_color[0]
                    G[temp4[0][0]][temp4[0][1]]['thickness'] = edge_thickness[0]

            if FCS_loc[i] == 1 and MCS_number[i] >= 1:
                count1 += 1
                count2 += 1
                temp1 = list(G.out_edges(num_nodes+count1))
                temp2 = list(G.in_edges(num_nodes+count1))
                temp3 = list(G.out_edges(num_nodes+count_FCS+count2))
                temp4 = list(G.in_edges(num_nodes+count_FCS+count2))

                if FCS_switch[od_pair_num + i*num_od_pair].X == 0\
                      and MCS_switch[od_pair_num + i*num_od_pair].X == 0\
                          and road_switch[od_pair_num + i*num_od_pair].X == 0:
                    G[temp1[0][0]][temp1[0][1]]['color'] = edge_color[0]
                    G[temp1[0][0]][temp1[0][1]]['thickness'] = edge_thickness[0]
                    G[temp2[0][0]][temp2[0][1]]['color'] = edge_color[0]
                    G[temp2[0][0]][temp2[0][1]]['thickness'] = edge_thickness[0]
                    G[temp3[0][0]][temp3[0][1]]['color'] = edge_color[0]
                    G[temp3[0][0]][temp3[0][1]]['thickness'] = edge_thickness[0]
                    G[temp4[0][0]][temp4[0][1]]['color'] = edge_color[0]
                    G[temp4[0][0]][temp4[0][1]]['thickness'] = edge_thickness[0]
                else:
                    G[temp1[0][0]][temp1[0][1]]['color'] = edge_color[1]
                    G[temp1[0][0]][temp1[0][1]]['thickness'] = edge_thickness[1]
                    G[temp2[0][0]][temp2[0][1]]['color'] = edge_color[1]
                    G[temp2[0][0]][temp2[0][1]]['thickness'] = edge_thickness[1]
                    G[temp3[0][0]][temp3[0][1]]['color'] = edge_color[1]
                    G[temp3[0][0]][temp3[0][1]]['thickness'] = edge_thickness[1]
                    G[temp4[0][0]][temp4[0][1]]['color'] = edge_color[1]
                    G[temp4[0][0]][temp4[0][1]]['thickness'] = edge_thickness[1]

    # empty_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr == {}]
    # print(empty_edges)
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_thicknesses = [G[u][v]['thickness'] for u, v in G.edges()]
        
    
    # node_labels = {list(pos_nodes.keys())[i]:list(pos_nodes.keys())[i] for i in range(num_nodes+count_FCS+count_MCS)}   # to label basic nodes, FCS and MCS 
    node_labels = nx.get_node_attributes(G, 'MCS_count')
    #############################################################################################################################################
    #### These labels are not automated
    #############################################################################################################################################
    # axs.text(-120.305, 37.138, f'Flow input:  {init_flow}', fontsize=20, color='black')
    # axs.text(-120.305, 37.098, 'FCS unused:', fontsize=20, color='black')
    # axs.scatter([-120.1698], [37.11], color='red', s=Node_size)
    # axs.text(-120.305, 37.07, 'FCS used:', fontsize=20, color='black')
    # axs.scatter([-120.1698], [37.082], color='green', s=Node_size)
    # axs.text(-120.305, 37.048, 'MCS unused:', fontsize=20, color='black')
    # axs.scatter([-120.1698], [37.06], color='yellow', s=Node_size)
    # axs.text(-120.305, 37.02, 'MCS used:', fontsize=20, color='black')
    # axs.scatter([-120.1698], [36.032], color='orange', s=Node_size)

    # axs.text(start[0], start[1], 'Start', fontsize=20, color='red')
    # axs.text(goal[0], goal[1], 'Goal', fontsize=20, color='green')

    ###########################################################################################################################################
    plt.margins(0.1)
    
    nx.draw(G,node_color=node_colors,node_size=node_sizes,pos=pos_nodes,edge_color=edge_colors,width=edge_thicknesses,ax=axs,
            with_labels=True,labels=node_labels,font_color="grey",font_size=7)
    # nx.draw(G,node_color=node_colors,node_size=node_sizes,pos=pos_nodes,edge_color=edge_colors,width=edge_thicknesses,ax=axs,
    #         with_labels=False)
    # axs.set_aspect('equal')
    # # Add labels, title, and legend
    # axs.set_xlim(-16, 28) 
    # axs.set_ylim(-23, 16) 
    # axs.set_title(f'OD pair: {od_pair_num+1}')

    return fig
    
    