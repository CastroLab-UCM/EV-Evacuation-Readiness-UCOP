"""Module for setting up SUMO simulations."""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import dataclasses
import os
import layer_lib
import constants
import traci.constants as tc
import configs
import json
import sumolib
from sumolib import checkBinary
import numpy as np
import pandas as pd


if configs.MODE == "gui":
    import traci
    SUMO_BINARY = checkBinary('sumo-gui')
else:
    import libsumo as traci
    SUMO_BINARY = checkBinary('sumo')


@dataclasses.dataclass
class FlowData():
    """All data that will be saved to a Excel file for each run."""
    # List of 2-tuple: 1st entry is station, 2nd entry is duration
    od_pair: tuple | None = None
    # the edge that goes into CS
    edge_to_stations: list | None = None
    flow_size: int | None = None
    path: str | None = None




@dataclasses.dataclass
class SumoOutputData():
    """All data that will be saved as output of sumo simulations."""
    # vehicle_travel_info has the following format:
    #   { veh_id: [start_ts, end_ts] }
    vehicle_travel_info: dict | None = dataclasses.field(default_factory=dict)
    edge_travel_info: dict | None = dataclasses.field(default_factory=dict)
    edge_traversal_count: dict | None = dataclasses.field(default_factory=dict)
    # Edge time tracking
    vehicle_current_edge: dict | None = dataclasses.field(default_factory=dict)
    vehicle_edge_enter_ts: dict | None = dataclasses.field(default_factory=dict)
    edge_time_spent: dict | None = dataclasses.field(default_factory=dict)
    # Parking (charging station) time tracking
    vehicle_parking_area: dict | None = dataclasses.field(default_factory=dict)   # {veh_id: parkingAreaId}
    vehicle_parking_enter_ts: dict | None = dataclasses.field(default_factory=dict)  # {veh_id: step}
    parking_area_time_spent: dict | None = dataclasses.field(default_factory=dict)   # {parkingAreaId: total_seconds}
    parking_area_visits: dict | None = dataclasses.field(default_factory=dict)       # {parkingAreaId: count}
    # Final energy at trip end
    vehicle_final_energy: dict | None = dataclasses.field(default_factory=dict)  # {veh_id: {"charge": float, "capacity": float}}
    # Cache of last-known energy for vehicles still on network
    vehicle_last_energy: dict | None = dataclasses.field(default_factory=dict)  # {veh_id: {"charge": float, "capacity": float}}


def build_kia_vehicle(root, vid, color):
    car = ET.SubElement(
        root,
        "vType",
        id=vid,
        minGap=constants.KIA_SOUR_EV_2020_CONSTRAINTS["minGap"],
        maxSpeed=constants.KIA_SOUR_EV_2020_CONSTRAINTS["maxSpeed"],
        color=color,
        accel=constants.KIA_SOUR_EV_2020_CONSTRAINTS["accel"],
        decel=constants.KIA_SOUR_EV_2020_CONSTRAINTS["decel"],
        sigma=constants.KIA_SOUR_EV_2020_CONSTRAINTS["sigma"],
        emissionClass=constants.KIA_SOUR_EV_2020_CONSTRAINTS["emissionClass"],
        mass=constants.KIA_SOUR_EV_2020_CONSTRAINTS["mass"],
    )
    for key, value in constants.KIA_SOUR_EV_2020_PARAMS.items():
        ET.SubElement(
            car,
            "param",
            key=key,
            value=value,
        )


def process_row(prev_row, current_row, edge_to_stations, path):
    if current_row['charged energy'] != 0:
        station_from_node = f"{int(current_row['i'])}"
        station_to_node = f"CS{int(current_row['i'])}"
        # station = f"STATION_{station_from_node}_{station_to_node}"
        energy = current_row['charged energy']
        stoptime = str(int((energy)/constants.RECHARGE_TIME_MILE_PER_S))
        if prev_row is None:
            edge_to_station = (None, int(current_row['i']))
        else:
            edge_to_station = (int(prev_row['i']), int(prev_row['j']))
        edge_to_stations.append([edge_to_station, stoptime])
        path += f"{station_from_node}_{station_to_node} "
        path += f"{station_to_node}_{station_from_node} "
    path += get_edge_from_row(current_row) + " "
    return path, edge_to_stations


def build_path_and_stations(data, od_pair):
    od_plan = data.od_plans[od_pair]
    path = ""
    edge_to_stations = []
    prev_row = None
    for _, current_row in od_plan.iterrows():
        path, edge_to_stations = process_row(prev_row, current_row, edge_to_stations, path)
        prev_row = current_row
    # from_row = od_plan.loc[od_plan.iloc[:, 0] == int(od_pair[0])]
    # to_row = od_plan.loc[od_plan.iloc[:, 1] == int(od_pair[1])]
    # current_row = from_row
    # path = ""
    # edge_to_stations = []
    # prev_row = None
    # while not current_row.equals(to_row):
    #     next_row = od_plan.loc[od_plan.loc[:, 'i'] == current_row['j'].values[0]]
    #     path, edge_to_stations = process_row(prev_row, current_row, edge_to_stations, path)
    #     else:
    #         break
    #     prev_row = current_row
    #     current_row = next_row
    # path, edge_to_stations = process_row(prev_row, current_row, edge_to_stations, path)
    # path = path + get_edge_from_row(to_row)
    return path, edge_to_stations


def get_edge_from_row(row):
    from_ = int(row['i'])
    to_ = int(row['j'])
    return f"e{from_}_{to_}"


def process_od_pair(data, od_pair):
    flow_data = FlowData()
    flow_data.od_pair = (od_pair[0], od_pair[1],od_pair[2])
    # Aggregate duplicates by (start_node, end_node, iteration)
    agg = (
        data.demanded_od_flow
        .groupby(["start_node", "end_node", "iteration"], as_index=True)["demanded_flow"]
        .sum()
    )
    flow_data.flow_size = int(agg.loc[flow_data.od_pair])
    # flow_data.flow_size = data.demanded_od_flow.set_index(["start_node", "end_node","iteration"]).loc[flow_data.od_pair]["demanded_flow"]
    flow_data.path, flow_data.edge_to_stations = build_path_and_stations(data, od_pair)
    return flow_data


def add_flow_to_root(root, flow_data, initial_energy):
    edges = flow_data.path.split(" ")
    from_edge = edges[0]
    to_edge = edges[-2]
    flow = ET.SubElement(
        root,
        "flow",
        id="-".join([str(i) for i in flow_data.od_pair]),
        begin=constants.SIMULATION_FLOW_BEGIN_TIME,
        end=configs.SIMULATION_FLOW_END_TIME,
        # number=str(flow_data.flow_size*configs.FLOW_SIMULATION_TIME),
        number=str(int(configs.DEMAND_SCALE*1000*configs.FLOW_SIMULATION_TIME)),
        from_=from_edge,
        to=to_edge,
        via=flow_data.path,
        type="EV2",
        timeToTeleport="-1"
    )
    ET.SubElement(
        flow,
        "param",
        key="device.battery.chargeLevel",
        value=initial_energy,
    )
    for edge_to_station, duration in flow_data.edge_to_stations:
        from_node, to_node = edge_to_station
        charging_station = f"STATION_{to_node}_CS{to_node}"
        parking_area = "P_" + charging_station
        if from_node is not None:
            ET.SubElement(
                flow,
                "stop",
                edge=f"e{from_node}_{to_node}",
                duration="0",
                jump="1"
            )
        ET.SubElement(
            flow,
            "stop",
            parkingArea=parking_area,
            chargingStation=charging_station,
            duration=duration,
            jump="1"
        )
        ET.SubElement(
            flow,
            "stop",
            edge=f"CS{to_node}_{to_node}",
            duration="0",
            jump="1"
        )
    return root


def write_xml(root, excel_file_path):
    xml_str = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")

    # Write the pretty-printed XML to a file
    with open(excel_file_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


def write_trips(data, output_dir, trips_file_name="trips.xml"):
    root = ET.Element("routes")
    build_kia_vehicle(root, "EV2", "white")
    for od_pair in data.od_plans.keys():
        flow_data = process_od_pair(data, od_pair)
        root = add_flow_to_root(root, flow_data, constants.INITIAL_ENERGY)

    excel_file_path = os.path.join(output_dir, trips_file_name)
    write_xml(root, excel_file_path)
    print(f"XML file {excel_file_path} created successfully with trips in the specified format.")
    return excel_file_path

def add_charging_station_and_parking_area(root, from_node, to_node):
    ET.SubElement(
        root,
        "parkingArea",
        id=f"P_STATION_{from_node}_{to_node}",
        lane=f"{from_node}_{to_node}_0",
        startPos="0",
        endPos="300",
        roadsideCapacity=str(configs.CHARGER_PROT_NUM),
        angle="45",
    )
    ET.SubElement(
        root,
        "chargingStation",
        id=f"STATION_{from_node}_{to_node}",
        lane=f"{from_node}_{to_node}_0",
        chargeDelay="0",
        chargeInTransit="0",
        power="50000",
        efficiency="1",
        startPos="0",
        endPos="300",
    )
    return root


def write_network(data,configs,output_dir,node_file_name="nodes.xml",edge_file_name = "edges.xml",net_file_name="net.xml"):
    charging_nodes = []
    for idx, row in data.charging_demand_size.iterrows():
        if row['facility'] > 0:
            charging_nodes.append(idx)
    original_node_path = configs.PATH_CONFIG.RELATIVE_NODE_XML_PATH
    original_edge_path = configs.PATH_CONFIG.RELATIVE_EDGE_XML_PATH

    new_node_path = os.path.join(output_dir, node_file_name)
    new_edge_path = os.path.join(output_dir, edge_file_name)
    new_net_path = os.path.join(output_dir, net_file_name)

    #copy original node and add stations, save to new directory
    tree = ET.parse(original_node_path)
    root = tree.getroot()
    for node in root.findall("node"):
        node_id = node.get("id")
        node.set("x",str(float(node.get("x"))*configs.MAP_SCALE))
        node.set("y",str(float(node.get("y"))*configs.MAP_SCALE))
        for station in charging_nodes:
            if node_id == str(station):
                # Add a new attribute to the node element
                charging_node = ET.SubElement(root,"node")
                charging_node.set("id", "CS"+str(station))
                charging_node.set("x",str(float(node.get("x"))+250))
                charging_node.set("y",str(float(node.get("y"))-250))
                charging_node.set("type", "allway_stop")
    write_xml(root, new_node_path)

    #copy original edge and add stations, save to new directory
    tree = ET.parse(original_edge_path)
    root = tree.getroot()

    for station in charging_nodes:
        charging_edge = ET.SubElement(root,"edge")
        charging_edge.set("id",str(station)+"_CS"+str(station))
        charging_edge.set("from",str(station))
        charging_edge.set("to","CS"+str(station))
        charging_edge.set("numLanes","4")

        charging_edge = ET.SubElement(root,"edge")
        charging_edge.set("id","CS"+str(station)+"_"+str(station))
        charging_edge.set("from","CS"+str(station))
        charging_edge.set("to",str(station))
        charging_edge.set("numLanes","2")
    write_xml(root, new_edge_path)

    print("new_node_path: ", new_node_path)
    print("new_edge_path: ", new_edge_path)
    print("new_net_path: ", new_net_path)
    status = os.system(f'netconvert --node-files={new_node_path} --edge-files={new_edge_path} --output-file={new_net_path}')

    print("status: ",status)
    return new_net_path



def write_charging_station_and_parking_area(data, configs, output_dir, add_file_name="add.xml"):
    charging_nodes = []
    for idx, row in data.charging_demand_size.iterrows():
        if row['facility'] >0:
            charging_nodes.append(idx)
    print("Charging nodes:", charging_nodes)

    root = ET.Element("routes")
    for station in charging_nodes:
        from_node = station
        to_node = "CS"+str(station)
        add_charging_station_and_parking_area(root, from_node, to_node)
        print(from_node, to_node)
    # edges, _, _ = layer_lib.read_graph(configs.RELATIVE_RAW_GRAPH_PATH)
    # for edge in edges:
    #     from_node, to_node = edge
    #     if from_node in charging_nodes:
    #         root = add_charging_station_and_parking_area(root, from_node, to_node)
    excel_file_path = os.path.join(output_dir, add_file_name)

    write_xml(root, excel_file_path)
    print(f"XML file {excel_file_path} created successfully with edges in the specified format.")
    return excel_file_path


def run_sumo_sim(net_file, route_file, add_file, view_file, data):
    edge_list = get_edge_list()
    startSim(net_file, route_file, add_file,view_file)
    step = 0
    output_data = SumoOutputData()
    # od_pairs = ['-'.join([str(station) for station in od_pair]) for od_pair in data.od_plans.keys()]
    while shouldContinueSim():
    #     # for vehId in getOurDeparted(VEHICLES):
    #     #     setVehColor(vehId, RED)
    #     #     avoidEdge(vehId, EDGE_ID)
        # print("step: ",step)
        traci.simulationStep()
        vehicles = traci.vehicle.getIDList()
        output_data = update_output_data(output_data, vehicles, step)

        # vehicle_on_edge = get_vehicle_on_edge("e0_1")
        if configs.FLOW_EDGE and (step % 3600 == 0):
            m = 61
            flow_size = np.zeros((m, m))
            for edge_id in edge_list:
                count = output_data.edge_traversal_count.get(edge_id, 0)
                if "CS" not in edge_id:
                    i = edge_id.split("_")[0][1:]
                    j = edge_id.split("_")[1]
                    flow_size[int(i)][int(j)] = count
            flow_size_pd = pd.DataFrame(flow_size)
            flow_output_path = os.path.join(os.path.dirname(configs.LAYER4_CONFIGS.result_dir), f'{step}.csv')
            flow_size_pd.to_csv(flow_output_path, index=False)
        step+=1
        # for i in range(len(vehicles)):
        #     vehid = vehicles[i]
            # traveled_distance = traci.vehicle.getDistance(vehid)
            # capacity = float(traci.vehicle.getParameter(vehid, "device.battery.capacity"))
            # currentCharge = float(traci.vehicle.getParameter(vehid, "device.battery.chargeLevel"))
            # stateOfCharge = currentCharge / capacity
            # consumed_energy=float(traci.vehicle.getParameter(vehid, "device.battery.totalEnergyConsumed"))
            # # print("consumed_energy",consumed_energy)
            # if traveled_distance<10000000:
            #     m_per_wh = 6.5
            # else:
            #     m_per_wh = traveled_distance /consumed_energy
            # # print("m_per_wh: ",m_per_wh)
            # remainingRange = float(traci.vehicle.getParameter(vehid, "device.battery.chargeLevel")) * m_per_wh
            # if currentCharge<=0:
            #     print("currentCharge: ",currentCharge)
    traci.close()
    m = 61
    flow_size = np.zeros((m, m))
    for edge_id in edge_list:
        count = output_data.edge_traversal_count.get(edge_id, 0)
        if "CS" not in edge_id:
            i = edge_id.split("_")[0][1:]
            j = edge_id.split("_")[1]
            flow_size[int(i)][int(j)] = count
    flow_szie_pd = pd.DataFrame(flow_size)
    flow_output_path = os.path.join(os.path.dirname(configs.LAYER4_CONFIGS.result_dir), f'{step}.csv')
    flow_szie_pd.to_csv(flow_output_path, index=False)


    print(f"Simulation ended at time {step}.")
    # print("Last seen od_pair timestamps: ", output_data.vehicle_travel_info)
    sumo_output_path = os.path.join(os.path.dirname(configs.LAYER4_CONFIGS.result_dir), 'SUMO_output.txt')
    with open(sumo_output_path, "w") as f:
        total_time = []
        for veh_id, value in output_data.vehicle_travel_info.items():
            start_time = value[0]
            end_time = value[1]
            travel_time = end_time - start_time
            total_time.append(travel_time)
            f.write(f"{veh_id}: start {start_time}, end {end_time}, travel {travel_time}\n")
        avg, std = np.mean(total_time), np.std(total_time)
        msg = f"Average travel time: {avg} +/- {std} seconds\n"
        f.write(msg)
        print(msg)

    # Average time on each edge to CS (edges containing "_CS")
    if output_data.edge_time_spent:
        cs_edges = {eid: t for eid, t in output_data.edge_time_spent.items() if "_CS" in eid}
        if cs_edges:
            print("Average time spent on edge-to-CS edges (seconds):")
            for eid, total_t in sorted(cs_edges.items()):
                traversals = output_data.edge_traversal_count.get(eid, 0)
                avg_t = (total_t / traversals) if traversals > 0 else 0
                print(f"  {eid}: {avg_t:.2f} (total={total_t}, traversals={traversals})")
        else:
            print("No edge-to-CS edges observed.")

    # Average time in corresponding charging stations (parking areas starting with 'P_STATION_')
    if output_data.parking_area_time_spent:
        print("Average time spent in charging stations (seconds):")
        for pid, total_t in sorted(output_data.parking_area_time_spent.items()):
            visits = output_data.parking_area_visits.get(pid, 0)
            avg_t = (total_t / visits) if visits > 0 else 0
            print(f"  {pid}: {avg_t:.2f} (total={total_t}, visits={visits})")

    # Print and append final energy levels for each vehicle
    if output_data.vehicle_final_energy:
        print("Final energy levels (Wh and % of capacity):")
        with open(sumo_output_path, "a") as f:
            f.write("Final energy levels (Wh and % of capacity):\n")
            for veh_id in sorted(output_data.vehicle_final_energy.keys()):
                info = output_data.vehicle_final_energy[veh_id]
                charge = info.get("charge") or 0.0
                cap = info.get("capacity") or 0.0
                pct = (charge / cap * 100.0) if cap else 0.0
                line = (
                    f"{veh_id}: charge={charge:.2f} Wh, capacity={cap:.2f} Wh, SoC={pct:.1f}%"
                )
                print(line)
                f.write(line + "\n")


def startSim(net_file, route_file, add_file,view_file):
    """Starts the simulation."""
    traci.start(
        [
            SUMO_BINARY,
            '--net-file', net_file,
            '--route-files', route_file,
            '--delay', '0',
            '--gui-settings-file', view_file,
            '--start',
            '--additional-files', add_file,
            # '--no-warnings','True'
            '--time-to-teleport','-1'

        ])


def shouldContinueSim():
    """Checks that the simulation should continue running.
    Returns:
        bool: `True` if vehicles exist on network. `False` otherwise.
    """
    numVehicles = traci.simulation.getMinExpectedNumber()
    if numVehicles == 0:
        print("No vehicles on network. Ending simulation.")
    return True if numVehicles > 0 else False


def setVehColor(vehId, color):
    """Changes a vehicle's color.
    Args:
        vehId (String): The vehicle to color.
        color ([Int, Int, Int]): The RGB color to apply.
    """
    traci.vehicle.setColor(vehId, color)


def getOurDeparted(filterIds=[]):
    """Returns a set of filtered vehicle IDs that departed onto the network during this simulation step.
    Args:
        filterIds ([String]): The set of vehicle IDs to filter for.
    Returns:
        [String]: A set of vehicle IDs.
    """
    newlyDepartedIds = traci.simulation.getDepartedIDList()
    filteredDepartedIds = newlyDepartedIds if len(
        filterIds) == 0 else set(newlyDepartedIds).intersection(filterIds)
    return filteredDepartedIds


def update_output_data(output_data, vehicles, step):
    output_data = update_vehicle_travel_info(output_data, vehicles, step)
    if configs.FLOW_EDGE:
        output_data = update_edge_travel_info(output_data, vehicles)
        output_data = update_edge_traversal_count(output_data, vehicles)
    output_data = update_edge_time_spent(output_data, vehicles, step)
    output_data = update_parking_time_spent(output_data, vehicles, step)
    output_data = update_vehicle_energy_cache(output_data, vehicles)
    output_data = update_vehicle_final_energy(output_data)
    return output_data


def update_vehicle_travel_info(output_data, vehicles, step):
    for vehicle in vehicles:
        if vehicle not in output_data.vehicle_travel_info:
            output_data.vehicle_travel_info[vehicle] = [step, step]
        else:
            output_data.vehicle_travel_info[vehicle][1] = step
    return output_data

def update_edge_travel_info(output_data, vehicles):
    for vehicle in vehicles:
        edge = traci.vehicle.getRoadID(vehicle)
        if edge not in output_data.edge_travel_info:
            output_data.edge_travel_info[edge] = {vehicle}
        else:
            output_data.edge_travel_info[edge].add(vehicle)
    return output_data

def update_edge_traversal_count(output_data, vehicles):
    for key in output_data.edge_travel_info.keys():
        output_data.edge_traversal_count[key] = len(output_data.edge_travel_info[key])
    return output_data

def update_vehicle_energy_cache(output_data, vehicles):
    """Update last-known energy values for vehicles currently on the network."""
    for veh in vehicles:
        # Best-effort queries; ignore failures
        try:
            charge = float(traci.vehicle.getParameter(veh, "device.battery.chargeLevel"))
        except Exception:
            charge = None
        try:
            capacity = float(traci.vehicle.getParameter(veh, "device.battery.capacity"))
        except Exception:
            capacity = None
        if charge is not None or capacity is not None:
            prev = output_data.vehicle_last_energy.get(veh, {})
            if charge is None:
                charge = prev.get("charge", 0.0)
            if capacity is None:
                capacity = prev.get("capacity", 0.0)
            output_data.vehicle_last_energy[veh] = {"charge": charge, "capacity": capacity}
    return output_data

def update_vehicle_final_energy(output_data):
    """Record final battery energy for vehicles that arrived this step."""
    try:
        arrived = traci.simulation.getArrivedIDList()
    except Exception:
        arrived = []
    for veh in arrived:
        if veh in output_data.vehicle_final_energy:
            continue
        # Use cached value only; do not query traci for arrived vehicles
        info = output_data.vehicle_last_energy.pop(veh, None)
        if info is None:
            info = {"charge": 0.0, "capacity": 0.0}
        output_data.vehicle_final_energy[veh] = info
    return output_data

def update_edge_time_spent(output_data, vehicles, step):
    """Track time spent per edge for each vehicle by detecting edge transitions."""
    for veh in vehicles:
        curr_edge = traci.vehicle.getRoadID(veh)
        prev_edge = output_data.vehicle_current_edge.get(veh)
        if prev_edge is None:
            output_data.vehicle_current_edge[veh] = curr_edge
            output_data.vehicle_edge_enter_ts[veh] = step
        elif prev_edge != curr_edge:
            enter_ts = output_data.vehicle_edge_enter_ts.get(veh, step)
            duration = max(0, step - enter_ts)
            output_data.edge_time_spent[prev_edge] = output_data.edge_time_spent.get(prev_edge, 0) + duration
            output_data.vehicle_current_edge[veh] = curr_edge
            output_data.vehicle_edge_enter_ts[veh] = step
    # Close segments for vehicles that arrived this step
    for veh in traci.simulation.getArrivedIDList():
        prev_edge = output_data.vehicle_current_edge.pop(veh, None)
        enter_ts = output_data.vehicle_edge_enter_ts.pop(veh, None)
        if prev_edge is not None and enter_ts is not None:
            duration = max(0, step - enter_ts)
            output_data.edge_time_spent[prev_edge] = output_data.edge_time_spent.get(prev_edge, 0) + duration
    return output_data

def update_parking_time_spent(output_data, vehicles, step):
    """Track time spent in parking areas (charging stations) by querying parking areas and their vehicle sets."""
    try:
        parking_ids = traci.parkingarea.getIDList()
    except Exception:
        parking_ids = []

    # Build current mapping of veh -> parking area (if any)
    current_parking = {}
    for pid in parking_ids:
        try:
            vehs_at_pid = traci.parkingarea.getVehicleIDs(pid)
        except Exception:
            vehs_at_pid = []
        for v in vehs_at_pid:
            current_parking[v] = pid

    # Update per-vehicle parking state
    for veh in vehicles:
        pid = current_parking.get(veh)
        prev_pid = output_data.vehicle_parking_area.get(veh)
        if pid is not None:
            if prev_pid is None:
                # vehicle just entered a parking area
                output_data.vehicle_parking_area[veh] = pid
                output_data.vehicle_parking_enter_ts[veh] = step
                output_data.parking_area_visits[pid] = output_data.parking_area_visits.get(pid, 0) + 1
            elif prev_pid != pid:
                # vehicle switched parking areas (close previous)
                enter_ts = output_data.vehicle_parking_enter_ts.get(veh, step)
                duration = max(0, step - enter_ts)
                output_data.parking_area_time_spent[prev_pid] = output_data.parking_area_time_spent.get(prev_pid, 0) + duration
                output_data.vehicle_parking_area[veh] = pid
                output_data.vehicle_parking_enter_ts[veh] = step
        else:
            # vehicle not parking now; if it was previously parking, close segment
            prev_pid = output_data.vehicle_parking_area.pop(veh, None)
            enter_ts = output_data.vehicle_parking_enter_ts.pop(veh, None)
            if prev_pid is not None and enter_ts is not None:
                duration = max(0, step - enter_ts)
                output_data.parking_area_time_spent[prev_pid] = output_data.parking_area_time_spent.get(prev_pid, 0) + duration

    # Close segments for vehicles that arrived this step
    for veh in traci.simulation.getArrivedIDList():
        prev_pid = output_data.vehicle_parking_area.pop(veh, None)
        enter_ts = output_data.vehicle_parking_enter_ts.pop(veh, None)
        if prev_pid is not None and enter_ts is not None:
            duration = max(0, step - enter_ts)
            output_data.parking_area_time_spent[prev_pid] = output_data.parking_area_time_spent.get(prev_pid, 0) + duration
    return output_data

def get_edge_list():
    original_edge_path = configs.PATH_CONFIG.RELATIVE_EDGE_XML_PATH
    net = sumolib.net.readNet(original_edge_path)
    edgeIDs = [e.getID() for e in net.getEdges()]
    return edgeIDs


def get_vehicle_on_edge(edge_id, vehicle_on_edge_old, count):
    vehicle_on_edge_new = traci.edge.getLastStepVehicleIDs(edge_id)
    for vehicleID in vehicle_on_edge_old:
        if vehicleID not in vehicle_on_edge_new:
            count+=1
    return vehicle_on_edge_new, count

