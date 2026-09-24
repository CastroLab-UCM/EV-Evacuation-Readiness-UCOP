import layer_lib
import configs
import os
import shutil
import time
import sumo_lib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
_CONFIG_PATH = os.path.join(configs.PYTHON_CODES_DIR, "od_model_slacks_flow_iep_fixcap_MP_charging/configs.py")
_CONSTANT_PATH = os.path.join(configs.PYTHON_CODES_DIR, "od_model_slacks_flow_iep_fixcap_MP_charging/constants.py")

def generate_sumo_files(data, result_dir, trips_file_name, add_file_name):
    trips_file_path = sumo_lib.write_trips(data, result_dir, trips_file_name)
    adds_file_path = sumo_lib.write_charging_station_and_parking_area(
        data, configs, result_dir, add_file_name
    )
    return trips_file_path, adds_file_path


def copy_config_file():
    destination_path = os.path.join(configs.BASE_DIR, 'config.py')
    shutil.copyfile(_CONFIG_PATH, destination_path)
    shutil.copyfile(_CONSTANT_PATH, destination_path)


def step_by_step_approach(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj, baseline,cap,capacity_info):
    facilities = layer_lib.layer2(infessible_od, infessible_od_demand, d_adj, graph,cap)
    data = layer_lib.layer3(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj, facilities, baseline,cap,capacity_info)
    net_file_path = sumo_lib.write_network(data,configs,configs.BASELINE_CONFIGS.result_dir,node_file_name="node.xml", edge_file_name="edge.xml",net_file_name="mariposa_small.net.xml")
    copy_config_file()
    trips_file_path =sumo_lib.write_trips(data, configs.LAYER3_CONFIGS.result_dir, trips_file_name="step_by_step.trips.xml")
    adds_file_path =sumo_lib.write_charging_station_and_parking_area(
        data, configs, configs.LAYER3_CONFIGS.result_dir, 'step_by_step.add.xml'
    )
    return net_file_path,trips_file_path, adds_file_path, data


def combined_apprach(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap,capacity_info):
    data = layer_lib.combined_layer4(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap,capacity_info)
    net_file_path = sumo_lib.write_network(data,configs,configs.LAYER4_CONFIGS.result_dir,node_file_name="node.xml", edge_file_name="edge.xml",net_file_name="mariposa_small.net.xml")
    copy_config_file()
    trips_file_path = sumo_lib.write_trips(
    data, configs.LAYER4_CONFIGS.result_dir, trips_file_name="combined_apprach.trips.xml"
    )
    adds_file_path = sumo_lib.write_charging_station_and_parking_area(
        data, configs, configs.LAYER4_CONFIGS.result_dir, 'combined_apprach.add.xml'
    )
    return net_file_path,trips_file_path, adds_file_path, data

def combined_apprach_with_flow_layer(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap,capacity_info):
    data = layer_lib.combined_layer5(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap,capacity_info)
    net_file_path = sumo_lib.write_network(data,configs,configs.BASELINE_CONFIGS.result_dir,node_file_name="node.xml", edge_file_name="edge.xml",net_file_name="mariposa_small.net.xml")
    trips_file_path = sumo_lib.write_trips(
        data, configs.LAYER5_CONFIGS.result_dir, trips_file_name="combined_apprach_with_flow_layer.trips.xml"
    )
    adds_file_path = sumo_lib.write_charging_station_and_parking_area(
        data, configs, configs.LAYER5_CONFIGS.result_dir, 'combined_apprach_with_flow_layer.add.xml'
    )
    return net_file_path,trips_file_path, adds_file_path, data






def baseline():
    output, data = layer_lib.baseline()
    write_sorted_time(data,layer=0)
    net_file_path = sumo_lib.write_network(data,configs,configs.BASELINE_CONFIGS.result_dir,node_file_name="node.xml", edge_file_name="edge.xml",net_file_name="mariposa_small.net.xml")
    trips_file_path = sumo_lib.write_trips(
        data, configs.BASELINE_CONFIGS.result_dir, trips_file_name="baseline.trips.xml"
    )
    adds_file_path = sumo_lib.write_charging_station_and_parking_area(
        data, configs, configs.BASELINE_CONFIGS.result_dir, 'baseline.add.xml'
    )
    return net_file_path,trips_file_path, adds_file_path, data

def tune():
    for map_weight in configs.MAP_WEIGHT_TUNE_VALUES:
        base_dir = os.path.join(configs.PATH_CONFIG.BASE_DIR, f'map_weight_{map_weight}')
        os.makedirs(base_dir)
        configs.MAP_WEIGHT = map_weight
        configs.PATH_CONFIG.base_dir = base_dir
        run_od()
        configs.PATH_CONFIG.base_dir = os.path.dirname(base_dir)

def run_baseline():
    net_file_path,trips_file_path, adds_file_path,data = baseline()
    copy_config_file()
    view_file_path = os.path.join(os.path.dirname(configs.PATH_CONFIG.RELATIVE_RAW_GRAPH_PATH), "viewSettings.xml")
    sumo_lib.run_sumo_sim(route_file=trips_file_path, add_file = adds_file_path,net_file=net_file_path,view_file=view_file_path, data=data)
    print("UEP finished successfully.")

def od_approach():
    solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap,capacity_info = layer_lib.layer1()
    facilities = np.zeros(len(graph.vs), dtype=int)
    # Set the i-th element in PRESET_STATION to 1
    for i in configs.PRESET_STATION:
        facilities[i] = 1
    sorted_od = layer_lib.sort_od_to_hazard(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph)
    data = None
    pre_charger_usage = np.zeros(len(graph.vs), dtype=int)
    old_od_pairs = None
    old_demand_list = None
    # new_cap = cap
    for key,value in sorted_od.items():
        if value == []:
            continue
        else:
            # if hasattr(data, 'flow_size'):
            #     new_cap = cap - (data.flow_size).to_numpy()
                # new_cap=np.where(new_cap < 0, 0, new_cap)
            # print(pd.DataFrame(new_cap))
            value_arr = np.array(value)
            od_pairs = value_arr[:, 0:3]
            demand_list = value_arr[:, 3]
            if not old_od_pairs is None:
                od_pairs = np.concatenate((old_od_pairs, od_pairs), axis=0)
                demand_list = np.concatenate((old_demand_list, demand_list), axis=0)
            if data is not None:
                facilities = data.charging_demand_size.iloc[:, 0].to_numpy()
                pre_charger_usage = data.charging_demand_size.iloc[:, 1].to_numpy()
            print("start solve layer 3")
            # print(od_pairs,demand_list)
            data = layer_lib.od_layer3(od_pairs, demand_list, graph, d_adj, facilities,pre_charger_usage=pre_charger_usage, cap=cap,capacity_info = capacity_info, data=data)
            # print((data.od_plans.values()))
            # for od_pair, od_plan in data.od_plans.items():
            #     print(f"OD Pair: {od_pair}, Plan: {od_plan}")
            old_od_pairs = od_pairs
            old_demand_list = demand_list

    write_sorted_time(data,layer=3)
    write_cost_file(data)
    net_file_path = sumo_lib.write_network(data,configs,configs.LAYER3_CONFIGS.result_dir,node_file_name="node.xml", edge_file_name="edge.xml",net_file_name="mariposa_small.net.xml")
    trips_file_path = sumo_lib.write_trips(data, configs.LAYER3_CONFIGS.result_dir, trips_file_name="od_apprach.trips.xml")
    adds_file_path = sumo_lib.write_charging_station_and_parking_area(
        data, configs, configs.LAYER3_CONFIGS.result_dir, 'od_apprach.add.xml'
    )
    return net_file_path,trips_file_path, adds_file_path, data

def write_cost_file(data):
    cost_file = os.path.join(configs.LAYER3_CONFIGS.result_dir, configs.LAYER3_CONFIGS.cost_output_excel_name)
    with pd.ExcelWriter(cost_file) as writer:
        for origin, cost in data.cost.items():
            cost.to_excel(writer, sheet_name=origin)

def write_sorted_time(data,layer):
    if layer == 3:
        time_file = os.path.join(configs.LAYER3_CONFIGS.result_dir, configs.LAYER3_CONFIGS.output_excel_name2)
    if layer == 4:
        time_file = os.path.join(configs.LAYER4_CONFIGS.result_dir, configs.LAYER4_CONFIGS.output_excel_name2)
    if layer == 0:
        time_file = os.path.join(configs.BASELINE_CONFIGS.result_dir, configs.BASELINE_CONFIGS.output_excel_name2)
    evacuation_time_df = data.evacuation_time

    # custom_order = [0,2,1,3,5,4]
    # custom_order = [0, 2, 1, 3, 5, 4, 11, 10, 8, 7, 9, 13, 15, 6, 16, 14, 17, 22, 18, 21, 20, 23, 19, 12]
    custom_order = [15,16,14]
    # Convert column 'a' to categorical type with the custom order
    evacuation_time_df['o'] = pd.Categorical(evacuation_time_df['o'], categories=custom_order, ordered=True)
    # Sort the DataFrame
    evacuation_time_sorted = evacuation_time_df.sort_values(by=['o', 'd'])
    with pd.ExcelWriter(time_file) as writer:
        evacuation_time_sorted.to_excel(writer)


def run_od():
    net_file_path,trips_file_path, adds_file_path,data = od_approach()
    copy_config_file()
    view_file_path = os.path.join(os.path.dirname(configs.PATH_CONFIG.RELATIVE_RAW_GRAPH_PATH), "viewSettings.xml")
    sumo_lib.run_sumo_sim(route_file=trips_file_path, add_file = adds_file_path,net_file=net_file_path,view_file=view_file_path, data=data)
    print("IEP approach finished successfully.")

def run_single():
    solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap,capacity_info = layer_lib.layer1()
    # #step_by_step_approach(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj, configs.BASELINE,cap)
    # trips_file_path, adds_file_path = combined_apprach_with_flow_layer(
    #     solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap
    # )
    print("start solve layer 4")
    net_file_path,trips_file_path, adds_file_path, data = combined_apprach(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap,capacity_info=capacity_info)
    write_sorted_time(data,layer = 4)
    copy_config_file()
    #baseline()
    # trips_file_path = "C:\\Users\\Shuang\\Box\\evacuation_data\\results\\20241124_1824\\SiouxFalls_layer5\\combined_apprach_with_flow_layer.add.xml"
    # adds_file_path = "C:\\Users\\Shuang\\Box\\evacuation_data\\results\\20241124_1824\\SiouxFalls_layer5\\combined_apprach_with_flow_layer.trips.xml"
    # net_file_path = "C:\\Users\\Shuang\\Documents\\GitHub\\ZEVEvacuation\\python_codes\\SiouxFalls\\siouxfalls.net.xml"
    view_file_path = os.path.join(os.path.dirname(configs.PATH_CONFIG.RELATIVE_RAW_GRAPH_PATH), "viewSettings.xml")
    # if configs.PLATFORM == 'Windows':
    sumo_lib.run_sumo_sim(route_file=trips_file_path, add_file = adds_file_path,net_file=net_file_path,view_file=view_file_path, data=data)
    print("JEP Finished successfully.")


def main():
    print("Starting...")
    if configs.LAYER == 'OPT':
        print("OPT...")
        run_od()
    elif configs.LAYER == 'BASELINE':
        run_baseline()
    elif configs.LAYER == 'SINGLE':
        run_single()
    else:
        raise Exception(f"Unsupported layer: {configs.LAYER}")

if __name__ == "__main__":
    main()
