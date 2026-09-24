from sys import platform
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import configs
import rules
import os
# from pyomo.environ import *
import openpyxl
import dataclasses
import random
from dataclasses import field
from random import seed
from random import randint
import time
import math
import csv
import constants
import copy
import collections
from decimal import Decimal

@dataclasses.dataclass
class ModelParams():
    graph: ig.Graph
    od_pairs: np.ndarray
    d_adj: np.ndarray
    demand_list: np.ndarray
    cap: np.ndarray
    capacity_info: np.ndarray
    od_plans: dict | None = field(default_factory=dict)
    facilities: np.ndarray | None = None
    layer: int = 1
    pre_charger_usage: np.ndarray | None = None


@dataclasses.dataclass
class RunData():
    """All data that will be saved to a Excel file for each run."""
    metadata: pd.DataFrame | None = None
    flow_size: pd.DataFrame | None = None
    actual_od_flow: pd.DataFrame | None = None
    demanded_od_flow: pd.DataFrame | None = None
    evacuation_time: pd.DataFrame | None = None
    charging_demand_size: pd.DataFrame | None = None
    lamb: pd.DataFrame | None = None
    od_plans: dict | None = field(default_factory=dict)
    cost: collections.OrderedDict | None = field(default_factory=collections.OrderedDict)
    model: pyo.ConcreteModel | None = None
    y: pd.DataFrame | None = None
    e: pd.DataFrame | None = None
    demand_size: pd.DataFrame | None = None

def random_existing_stations(start: int, end: int, num_to_generate: int = None) -> int:
    """
    Randomly select one integer from a specified range [start, end].

    Args:
        start (int): The start of the range (inclusive).
        end (int): The end of the range (inclusive).
        seed (int, optional): The seed for the random number generator.

    Returns:
        list[int]: A randomly selected integer from the range.
    """
    random.seed(15)
    return random.sample(range(start, end+1), num_to_generate)


def init_model(model_params: ModelParams):
    model = pyo.ConcreteModel()
    model.graph = model_params.graph
    model.demand_list = model_params.demand_list
    # edge weights
    model.d_adj = model_params.d_adj
    # OD pair
    model.OD = model_params.od_pairs
    model.nodes_size = model_params.graph.vcount()
    # all nodes
    model.nodes_set = pyo.RangeSet(0, model.nodes_size - 1)
    od_size = len(model.OD)
    model.OD_set = pyo.RangeSet(0, od_size - 1)
    # FCS
    model.o = np.zeros(model.nodes_size, dtype=int)
    # exist_station = make_random_station(model.nodes_size, configs.COUNT_EXISTING_STATIONS)
    exist_station = configs.PRESET_STATION
    for i in exist_station:
        model.o[i] = 1
    # previously solved stations (FCS + MCS)
    model.pre_f = copy.deepcopy(model.o)
    # decision variable for route
    model.x = pyo.Var(model.nodes_set, model.nodes_set, model.OD_set, within=pyo.Binary)
    # decision variable for recharge
    model.y = pyo.Var(model.nodes_set, model.OD_set, within=pyo.Binary)
    # Refueled at node i on a forward path of O-D pair od
    model.e = pyo.Var(model.nodes_set, model.OD_set, domain=pyo.NonNegativeIntegers)
    # Fuel remaining with arrival at node i on a forward path of O-D pair od
    model.r = pyo.Var(model.nodes_set, model.OD_set, domain=pyo.NonNegativeIntegers)
    # auxiliary variable
    model.a = pyo.Var(model.nodes_set, model.OD_set, domain=pyo.NonNegativeIntegers)
    #model.BPR = pyo.Var(model.nodes_set, model.nodes_set, domain=pyo.NonNegativeIntegers)
    model.V = constants.VEHICLE_DRIVING_RANGE
    # B=1e9;A large value
    model.B = constants.VERY_LARGE_NUMBER
    # recharge time per unit travelable distance
    model.t = constants.RECHARGE_TIME
    model.capacity_info = model_params.capacity_info
    model.od_plans = model_params.od_plans
    # Layer specific variable initialization
    if model_params.layer == 1:
        model.f = model.o
    if model_params.layer == 2:
        model.f = pyo.Var(model.nodes_set, within=pyo.Binary)
        model.s = constants.DRIVING_SPEED
        model.cap = model_params.cap
        model.charger_usage = pyo.Var(model.nodes_set, domain=pyo.NonNegativeIntegers)
        # try
        #model.max_road_flow = pyo.Var(
        #    model.nodes_set, model.nodes_set, domain=pyo.NonNegativeIntegers, bounds=(0, constants.TOPX)
        #)
        #model.BPR_SLACK = pyo.Var(model.nodes_set, model.nodes_set, domain=pyo.NonNegativeReals)
    if model_params.layer == 3:
        if model_params.pre_charger_usage is None:
            model.pre_charger_usage = np.zeros(model.nodes_size, dtype=int)
        else:
            model.pre_charger_usage = model_params.pre_charger_usage
        # Import solution from layer 2
        model.f = pyo.Var(model.nodes_set, within=pyo.Binary)
        for idx, value in enumerate(model_params.facilities):
            if value >0:
                model.pre_f[idx] = 1
        model.s = constants.DRIVING_SPEED
        model.charger_usage = pyo.Var(model.nodes_set, domain=pyo.NonNegativeIntegers)
        model.cap = model_params.cap
        # road flow
        model.max_road_flow = pyo.Var(
            model.nodes_set, model.nodes_set, domain=pyo.NonNegativeIntegers, bounds=(0, constants.TOPX)
        )
        # model.BPR_SLACK = pyo.Var(domain=pyo.NonNegativeReals)
        model.BPR_SLACK = pyo.Var(model.nodes_set, model.nodes_set, domain=pyo.NonNegativeReals)
        # model.BPR_SLACK_OD = pyo.Var(model.nodes_set, model.nodes_set,model.OD_set, domain=pyo.NonNegativeReals)
        # model.n = pyo.Var(model.OD_set, domain=pyo.NonNegativeIntegers)
        model.n = model.demand_list
        model.CS_SLACK = pyo.Var(model.nodes_set,domain=pyo.NonNegativeReals)
        model.CS_SLACK_OD = pyo.Var(model.nodes_set,model.OD_set,domain=pyo.NonNegativeReals)
        # model.BPR = pyo.Var(model.nodes_set, model.nodes_set, domain=pyo.NonNegativeReals)
        # model.road_flow_breakpt = [i * (constants.TOPX // configs.N_BREAKPOINTS) for i in range(configs.N_BREAKPOINTS + 1)]
    if model_params.layer == 4:
        model.f = pyo.Var(model.nodes_set, within=pyo.Binary)
        model.s = constants.DRIVING_SPEED
        model.n = model.demand_list
        model.cap = model_params.cap
        model.charger_usage = pyo.Var(model.nodes_set, domain=pyo.NonNegativeIntegers)
        # try
        model.max_road_flow = pyo.Var(
            model.nodes_set, model.nodes_set, domain=pyo.NonNegativeIntegers, bounds=(0, constants.TOPX)
        )
        # model.BPR_SLACK = pyo.Var(domain=pyo.NonNegativeReals)
        model.BPR_SLACK = pyo.Var(model.nodes_set, model.nodes_set, domain=pyo.NonNegativeReals)
        # model.BPR_SLACK_OD = pyo.Var(model.nodes_set, model.nodes_set,model.OD_set, domain=pyo.NonNegativeReals)
        model.CS_SLACK = pyo.Var(model.nodes_set,domain=pyo.NonNegativeReals)
        model.CS_SLACK_OD = pyo.Var(model.nodes_set,model.OD_set,domain=pyo.NonNegativeReals)

    if model_params.layer == 5:
        model.f = pyo.Var(model.nodes_set, within=pyo.Binary)
        model.s = constants.DRIVING_SPEED
        model.cap = model_params.cap
        model.charger_usage = pyo.Var(model.nodes_set, domain=pyo.NonNegativeIntegers)
        # try
        model.max_road_flow = pyo.Var(
            model.nodes_set, model.nodes_set, domain=pyo.NonNegativeIntegers, bounds=(0, constants.TOPX)
        )
        model.BPR_SLACK = pyo.Var(model.nodes_set, model.nodes_set, domain=pyo.NonNegativeReals)
        model.n = pyo.Var(model.OD_set, domain=pyo.NonNegativeIntegers)
    return model


def formulate_optimization_problem(model, layer, output_dir=None):
    # Remember to modify LayerConfigs.output_model_constraints as you modify the constraint here.
    model.openedge = pyo.Constraint(model.nodes_set, model.nodes_set, model.OD_set, rule=rules.openedge_)
    model.flow = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.flow_)
    model.forward_lower = pyo.Constraint(
        model.nodes_set, model.nodes_set, model.OD_set, rule=rules.forward_lower_
    )
    model.forward_upper = pyo.Constraint(
        model.nodes_set, model.nodes_set, model.OD_set, rule=rules.forward_upper_
    )
    model.refuel_decision = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.refuel_decision_)
    model.refuel_constraint = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.refuel_constraint_)
    model.init_fuel = pyo.Constraint(model.OD_set, rule=rules.init_fuel_)
    model.can_refuel = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.can_refuel_)
    model.noloop = pyo.Constraint(model.nodes_set, model.OD_set,rule=rules.noloop_)

    if layer == 1:
        model.objective = pyo.Objective(rule=rules.layer1_cost_)
        model.facility = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.layer12_facility_)
    if layer == 2:
        model.objective = pyo.Objective(rule=rules.layer2_cost_)
        model.facility = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.layer12_facility_)
        model.exist_facility = pyo.Constraint(model.nodes_set, rule=rules.exist_facility_)
        model.max_charger_usage = pyo.Constraint(model.nodes_set, rule=rules.layer2_max_charger_usage_)
        # model.charger_usage_const = pyo.Constraint(model.nodes_set, rule=rules.charger_usage_const_)
        # Experiments
        if configs.FACILITY_SUM_CONSTRAINT:
            model.facility_sum = pyo.Constraint(rule=rules.facility_sum_)
        # model.BPR_constraint = pyo.Constraint(model.nodes_set, model.nodes_set, rule=rules.BPR_constraint_)
        # model.max_road_flow_constraint = pyo.Constraint(model.nodes_set, model.nodes_set, rule=rules.max_road_flow_constraint_)
    if layer == 3:
        model.objective = pyo.Objective(rule=rules.layer3_cost_)
        model.facility = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.layer3_facility_)
        model.max_charger_usage = pyo.Constraint(model.nodes_set, rule=rules.max_charger_usage_layer3_)
        model.charger_usage_const = pyo.Constraint(model.nodes_set, rule=rules.charger_usage_const_layer3_)
        # Experiments
        model.max_road_flow_constraint = pyo.Constraint(model.nodes_set, model.nodes_set, rule=rules.max_road_flow_constraint_)
        model.BPR_constraint = pyo.Constraint(model.nodes_set, model.nodes_set, rule=rules.BPR_constraint_)
        model.exist_facility = pyo.Constraint(model.nodes_set, rule=rules.exist_facility_)
        if configs.FACILITY_SUM_CONSTRAINT:
            model.facility_sum_max = pyo.Constraint(rule=rules.facility_sum_max_)
        model.big_M_CS_constraint_lower = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.big_M_CS_constraint_lower_)
        model.big_M_CS_constraint_upper = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.big_M_CS_constraint_upper_)
        model.fix_od_constraint = pyo.Constraint(model.nodes_set,model.nodes_set,model.OD_set, rule=rules.fix_od_constraint_)
    if layer == 4:
        model.objective = pyo.Objective(rule=rules.layer4_cost_)
        model.facility = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.layer12_facility_)
        model.max_charger_usage = pyo.Constraint(model.nodes_set, rule=rules.max_charger_usage_layer4_)
        model.charger_usage_const = pyo.Constraint(model.nodes_set, rule=rules.charger_usage_const_layer4_)
        model.exist_facility = pyo.Constraint(model.nodes_set, rule=rules.exist_facility_)
        # Experiments
        model.max_road_flow_constraint = pyo.Constraint(model.nodes_set, model.nodes_set, rule=rules.max_road_flow_constraint_)
        model.BPR_constraint = pyo.Constraint(model.nodes_set, model.nodes_set, rule=rules.BPR_constraint_)
        if configs.FACILITY_SUM_CONSTRAINT:
            model.facility_sum_max = pyo.Constraint(rule=rules.facility_sum_max_)
        model.big_M_CS_constraint_lower = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.big_M_CS_constraint_lower_)
        model.big_M_CS_constraint_upper = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.big_M_CS_constraint_upper_)
    if layer == 5:
        model.objective = pyo.Objective(rule=rules.layer5_cost_)
        model.facility = pyo.Constraint(model.nodes_set, model.OD_set, rule=rules.layer12_facility_)
        model.exist_facility = pyo.Constraint(model.nodes_set, rule=rules.exist_facility_)
        model.max_charger_usage = pyo.Constraint(model.nodes_set, rule=rules.layer2_max_charger_usage_)
        model.charger_usage_const = pyo.Constraint(model.nodes_set, rule=rules.charger_usage_const_layer5_)
        # Experiments
        model.max_road_flow_constraint = pyo.Constraint(model.nodes_set, model.nodes_set, rule=rules.max_road_flow_constraint_layer5_)
        model.BPR_constraint = pyo.Constraint(model.nodes_set, model.nodes_set, rule=rules.BPR_constraint_)
        if configs.FACILITY_SUM_CONSTRAINT:
            model.facility_sum = pyo.Constraint(rule=rules.facility_sum_)

    if output_dir is not None:
        layer_config = configs.LAYER_CONFIGS[layer]
        if layer_config.output_model_constraints is not None and configs.SAVE_RESULTS:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for output_constraint in layer_config.output_model_constraints:
                if not os.path.exists(os.path.join(output_dir, 'debug_constraints')):
                    os.makedirs(os.path.join(output_dir, 'debug_constraints'))
                file_path = os.path.join(output_dir, 'debug_constraints', f'{output_constraint}.txt')
                with open(file_path, 'w') as output_file:
                    getattr(model, output_constraint).pprint(output_file)
    return model

def write_log_file(solver, layer):
    config = configs.LAYER_CONFIGS[layer]
    solver.options["logfile"] = os.path.join(config.result_dir, f"layer{layer}_solver.log")


def solve_model(model, debug, layer):
    solver = pyo.SolverFactory('gurobi')
    solver.options["MIPFocus"] = constants.MIP_FOCUS
    solver.options["MIPGap"]=constants.MIP_GAP
    solver.options["Cuts"]=constants.CUTS
    if layer not in [0, 1]:
        write_log_file(solver, layer)
    results = solver.solve(model,tee=debug)
    if (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ):
        if debug:
            print("this is feasible and optimal")
            print(results)
        return True
    elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        if debug:
            print("infeasible")
    else:
        if debug:
        # something else is wrong
            print("something else is wrong")
    return False


def save_charging_demand_size(model, data, debug, layer):
    charging_demand_size = np.zeros([model.nodes_size])
    facility = np.zeros([model.nodes_size])
    for i in model.nodes_set:
        if layer == 0:
            charging_demand_size[i]=(sum(model.y[i, k]*model.demand_list[k] for k in model.OD_set))
        elif layer == 3 or layer == 5:
            # charging_demand_size[i]=model.charger_usage[i].value
            charging_demand_size[i]=(sum(model.y[i, k].value*model.demand_list[k] for k in model.OD_set))
        else:
            # charging_demand_size[i]=model.charger_usage[i].value
            charging_demand_size[i]=(sum(model.y[i, k].value*model.demand_list[k] for k in model.OD_set))
            #charging_demand_size[i]=(sum(model.e[i, k].value*model.demand_list[k] for k in model.OD_set))
        if layer == 2 or layer == 4 or layer ==5 or layer ==3:
            facility[i] = model.f[i].value
        else:
            facility[i] = model.f[i]
        if debug:
            print(charging_demand_size[i])

    df_charging_demand_size = pd.DataFrame({
        "facility": facility.tolist(),
        "charging demand size": charging_demand_size.tolist(),
    })

    if data.charging_demand_size is None:
        data.charging_demand_size = df_charging_demand_size
    else:
        data.charging_demand_size = data.charging_demand_size + df_charging_demand_size

def save_y_e_demand(model,data,layer):
    y = np.zeros([model.nodes_size, len(model.demand_list)])
    e = np.zeros([model.nodes_size, len(model.demand_list)])
    demand_size = np.zeros([len(model.demand_list)])
    if layer == 0:
        data.y = model.y
        data.e = model.e
        data.demand_size = model.demand_list
    else:
        for i in model.nodes_set:
            for k in model.OD_set:
                y[i,k] = model.y[i,k].value
                e[i,k] = model.e[i,k].value
        demand_size = model.demand_list
        if data.y is None:
            data.y = y
            data.e = e
            data.demand_size = demand_size
        else:
            data.y = np.concatenate((data.y, y), axis=1)
            data.e = np.concatenate((data.e, e), axis=1)
            data.demand_size = np.concatenate((data.demand_size, demand_size),axis=0)
    return data





def save_average_wait_time(model, data, debug, layer):
    lamb = np.zeros([model.nodes_size])
    mu = np.zeros([model.nodes_size])
    t_ch = np.zeros([model.nodes_size])
    rho = np.zeros([model.nodes_size])
    p0 = np.zeros([model.nodes_size])
    t_wait = np.zeros([model.nodes_size],dtype=Decimal)
    term1 = np.zeros([model.nodes_size],dtype=Decimal)
    term2 = np.zeros([model.nodes_size],dtype=Decimal)
    term3 = np.zeros([model.nodes_size],dtype=Decimal)
    c = configs.CHARGER_PROT_NUM
    data.demand_size = data.demand_size.ravel()
    OD_size = int(data.demand_size.size)
    for i in model.nodes_set:
        lamb[i]=sum(data.y[i, k]*data.demand_size[k] for k in range(OD_size))
    for i in model.nodes_set:
        if lamb[i] > 0:
            t_ch[i] = sum((data.demand_size[k]*data.e[i,k]*constants.RECHARGE_TIME)/lamb[i] for k in range(OD_size))
            if t_ch[i] > 0:
                mu[i] = 1/t_ch[i]
            else:
                mu[i] = None
        else:
            t_ch[i] = None
            mu[i] = None

    r = lamb/mu
    rho = lamb/(c*mu)
    # print("__________rho:",rho)
    c_factorial= (math.factorial(c))
    for i in model.nodes_set:
        if not math.isnan(r[i]):
            term1[i] = (r[i]**c)/(c_factorial*(1-rho[i]))
            term2[i] = 0
            for n in range(c):
                term2[i] += ((r[i]**n)/(math.factorial(n)))
            term3[i] = (term1[i])/(c*mu[i]*(1-rho[i]))
            p0[i] = 1/(term1[i] + term2[i])
            t_wait[i] = (term3[i])*(p0[i])
    # print("mu[4]:",mu[4])
    # print("lamb[4]:",lamb[4])
    # print("rho[4]:",rho[4])
    # print("t_wait:",t_wait)
    data.average_wait_time = t_wait
    return data



def save_flow_size(model, data, debug, layer):
    flow_size = np.zeros([model.nodes_size, model.nodes_size])
    if layer == 2 or layer == 4:
        for i in model.nodes_set:
            for j in model.nodes_set:
                if model.cap[i,j]!=0:
                    flow_size[i, j] = sum(model.x[i, j, k].value * model.demand_list[k] for k in model.OD_set)+configs.EXIST_FLOW

    elif layer == 5 or layer == 3:
        for i in model.nodes_set:
            for j in model.nodes_set:
                if model.cap[i,j]!=0:
                    if not np.any(model.pre_charger_usage):
                        flow_size[i, j] = sum(model.x[i, j, k].value * model.demand_list[k] for k in model.OD_set)+configs.EXIST_FLOW
                    else:
                        flow_size[i, j] = sum(model.x[i, j, k].value * model.demand_list[k] for k in model.OD_set)

    else:
        for i in model.nodes_set:
            for j in model.nodes_set:
                if model.cap[i,j]!=0:
                    flow_size[i, j] = sum(model.x[i, j, k] * model.n[k][0] for k in model.OD_set)+configs.EXIST_FLOW
    if debug:
        print(flow_size[i, j])
    df_flow_size = pd.DataFrame(flow_size)
    data.flow_size = df_flow_size
    # if data.flow_size is None:
    #     data.flow_size = df_flow_size
    # else:
    #     data.flow_size += df_flow_size

    nonzero_flow_size = [n for n in flow_size.flatten() if n != 0]
    metadata = pd.DataFrame({
        "name": ['non-zero flow_size mean', 'non-zero flow_size std'],
        "value": [np.mean(nonzero_flow_size), np.std(nonzero_flow_size)],
    })
    data.metadata = metadata

    # TODO: Merge actual_flow and demanded_flow into one DataFrame
    if layer == 5:
        actual_flow = pd.DataFrame(model.n[k].value for k in model.OD_set)
        OD = pd.DataFrame(model.OD[k] for k in model.OD_set)
        actual_flow = pd.concat([OD, actual_flow], axis=1)
        demanded_flow = pd.DataFrame(model.demand_list[k] for k in model.OD_set)
        demanded_flow = pd.concat([OD, demanded_flow], axis=1)
    elif layer == 0 or layer == 4 or layer == 3:
        OD = pd.DataFrame(model.OD[k] for k in model.OD_set)
        # actual_flow = pd.DataFrame(model.n[k] for k in model.OD_set)
        demanded_flow = pd.DataFrame(model.demand_list[k] for k in model.OD_set)
        demanded_flow = pd.concat([OD, demanded_flow], axis=1)
        actual_flow = demanded_flow
    actual_flow.columns = ["start_node", "end_node", "iteration", "actual_flow"]
    demanded_flow.columns = ["start_node", "end_node", "iteration", "demanded_flow"]

    if data.actual_od_flow is None:
        data.actual_od_flow = actual_flow
        data.demanded_od_flow = demanded_flow
    else:
        data.actual_od_flow = pd.concat([data.actual_od_flow, actual_flow], axis=0)
        data.demanded_od_flow = pd.concat([data.demanded_od_flow, demanded_flow], axis=0)


def traverse_node(model, k, debug,layer):
    curr = model.OD[k][0]
    while curr != model.OD[k][1]:
        for v in model.nodes_set:
            if layer == 0:
                if round(model.x[curr, v, k]) != 1:
                    continue
            else:
                if round(model.x[curr, v, k].value) != 1:
                    continue
            # print(curr, "-->", v)
            # if debug:
            #     print(curr, "-->", v)
            curr = v
            break


def get_solution(model, k,layer,data):
    drive_time_BPR = 0
    wait_time = 0
    if layer == 0 or layer ==3 or layer == 4 or layer == 5:
        for i in model.nodes_set:
            for j in model.nodes_set:
                if model.cap[i,j]!=0:
                    if layer == 0:
                        drive_time_BPR += (model.d_adj[i, j] / model.s * model.x[i, j, k]*(1 + 0.15 * (data.flow_size.iloc[i,j]/model.capacity_info[i,j]) ** 4))
                    else:
                        drive_time_BPR += (model.d_adj[i, j] / model.s * model.x[i, j, k].value*(1 + 0.15 * (data.flow_size.iloc[i][j]/model.capacity_info[i,j]) ** 4))
                else:
                    continue
        drive_time_BPR = drive_time_BPR / 60
    if layer == 0:
        drive_time = (
            sum(
                model.d_adj[i, j] / model.s * model.x[i, j, k]
                for i in model.nodes_set
                for j in model.nodes_set
            )
            / 60
        )
        charge_time = sum(model.e[i, k] * model.t for i in model.nodes_set)
        wait_time = sum(model.y[i,k]*data.average_wait_time[i] for i in model.nodes_set)
    else:
        drive_time = (
            sum(
                model.d_adj[i, j] / model.s * model.x[i, j, k].value
                for i in model.nodes_set
                for j in model.nodes_set
            )
            / 60
        )
        charge_time = sum(model.e[i, k].value * model.t for i in model.nodes_set)
        wait_time = sum(model.y[i,k].value * data.average_wait_time[i] for i in model.nodes_set)
    df_series = pd.DataFrame(
        {
            "OD pair": [model.OD[k]],
            "o": model.OD[k][0],
            "d": model.OD[k][1],
            "drive time[h]": [drive_time],
            "drive time bpr[h]": [drive_time_BPR],
            "charge time [h]": [charge_time],
            "charge wait time [h]": [wait_time],
            "total time [h]": [drive_time_BPR + charge_time + wait_time],
        }
    )
    df_series.index = [int(k)]
    return df_series


def build_route_and_charging(model, k,layer):
    df_route_and_charging = pd.DataFrame({"i": [], "j": []})
    for e in model.graph.es:
        i = e.tuple[0]
        j = e.tuple[1]
        if layer == 0:
            round_route = round(model.x[i, j, k])
            remaining_energy = round(model.r[i, k])
            charged_energy = round(model.e[i, k])
        else:
            round_route = round(model.x[i, j, k].value)
            remaining_energy = round(model.r[i, k].value)
            charged_energy = round(model.e[i, k].value)
        if round_route == 1:
            e["route"] = True
            df_route_ij = pd.DataFrame({"i": [i], "j": [j], "remaining energy": [remaining_energy], "charged energy": [charged_energy]})
            df_route_and_charging = pd.concat([df_route_and_charging, df_route_ij], axis=0)
        else:
            e["route"] = False
    return df_route_and_charging


def get_node_color_mode(model, k, layer):
    for v in model.graph.vs:
        i = v.index
        fi = model.f[i].value if (layer == 2 or layer ==4 or layer == 5) else model.f[i]
        yi = model.y[i, k] if layer == 0 else model.y[i, k].value
        # start node
        if i == model.OD[k][0]:
            v["charge"] = 0
            v["shape"] = 0
        # end node
        elif i == model.OD[k][1]:
            v["charge"] = 4
            v["shape"] = 0
        # charging action
        elif yi == 1:
            v["shape"] = 1
            if model.o[i] == 1:
                v["charge"] = 1
            elif fi - model.o[i] == 1:
                v["charge"] = 2
        elif model.o[i] == 1:
            v["charge"] = 1
            v["shape"] = 0
        elif fi - model.o[i] == 1:
            v["charge"] = 2
            v["shape"] = 0
        else:
            v["charge"] = 3
            v["shape"] = 0
    return model


def set_visual_style(model, k, layer):
    color_dict = ["yellow", "pink", "red", "grey", "blue"]

    shape_dict = ["circle", "rectangle"]
    visual_style = {}

    visual_style["vertex_size"] = 20
    model = get_node_color_mode(model, k, layer)
    visual_style["vertex_color"] = [color_dict[charge] for charge in model.graph.vs["charge"]]

    visual_style["vertex_shape"] = [shape_dict[shape] for shape in model.graph.vs["shape"]]
    visual_style["vertex_label"] = [str(i) for i in model.nodes_set]
    visual_style["edge_label"] = model.graph.es["weight"]
    visual_style["edge_width"] = [2 + 8 * int(route) for route in model.graph.es["route"]]
    visual_style["bbox"] = (10000, 10000)
    visual_style["margin"] = 20
    return visual_style


def handle_repeated_nodes(route, repeating_node, sorted_route):
    # Find the back-and-forth edge
    for to_node in route[route['i'] == repeating_node]['j'].tolist():
        if not route[route['i'] == to_node][route['j'] == repeating_node].empty:
            to_charging_station_edge = route[
                (route['i'] == repeating_node) & (route['j'] == to_node)
            ]
            from_charging_station_edge = route[
                (route['i'] == to_node) & (route['j'] == repeating_node)
            ]
            sorted_route = pd.concat(
                [sorted_route, to_charging_station_edge, from_charging_station_edge], axis=0
            )
            route = route.drop(
                index=[to_charging_station_edge.index[0], from_charging_station_edge.index[0]]
            )
            break
    return route, sorted_route


def sort_route(od_pair, route):
    """Sort routes dataframe such that the first row is the start row and the last row is the end row."""
    # Check if the same node shows up as the start node - this would mean
    # the car take a detour to another station for charging.
    repeated_nodes = set(
        node for node in route['i'] if route['i'].tolist().count(node) > 1
    )
    from_node, to_node, _ = od_pair
    cur_node = from_node
    sorted_route = pd.DataFrame()
    while cur_node != to_node:
        if cur_node in repeated_nodes:
            route, sorted_route = handle_repeated_nodes(route, cur_node, sorted_route)
            repeated_nodes.remove(cur_node)
            continue
        cur_row = route[route['i'] == cur_node]
        sorted_route = pd.concat([sorted_route, cur_row], axis=0)
        next_node = cur_row['j'].values[0]
        cur_node = next_node
    return sorted_route.reset_index(drop=True)


def save_model(model, data, k, debug, layer):
    OD_pair = model.OD[k]
    df_route_and_charging = sort_route(
        OD_pair, build_route_and_charging(model, k,layer).reset_index(drop=True)
    )
    if debug:
        print(df_route_and_charging)

    data.od_plans.update({
        (OD_pair[0], OD_pair[1], OD_pair[2]): df_route_and_charging
    })


def df_result(model, data, debug, layer):
    if debug:
        for i in model.nodes_set:
            fi = model.f[i].value if layer == 2 else model.f[i]
            print("f: ", fi)
    df_sol = pd.DataFrame()
    for k in model.OD_set:
        if debug:
            print("OD: ", model.OD[k])
        # traverse_node(model, k, debug=debug,layer = layer)
        df_series = get_solution(model, k,layer,data)
        df_sol = pd.concat([df_sol, df_series], axis=0)
        save_model(model, data, k, debug, layer)
    data.evacuation_time = df_sol
    # if data.evacuation_time is None:
    #     data.evacuation_time = df_sol
    # else:
    #     data.evacuation_time = pd.concat([data.evacuation_time, df_sol], axis=0)
    if debug:
        print(df_sol)


#######################
## Layer 1 functions
#######################
# read graph from csv
def read_graph(graph_csv_file_path):
    df = pd.read_csv(graph_csv_file_path, dtype=str)
    edges_info = df["edge"].tolist()
    edges = []
    for e in edges_info:
        e = e.split(".")
        i = int(e[0]) - 1
        j = int(e[1]) - 1
        edges.append([i, j])
    weight_info = df["Length"].tolist()
    # weights = [int(configs.MAP_SCALE*float(ele)) for ele in weight_info]
    weights = [math.ceil(configs.MAP_SCALE*float(ele)) for ele in weight_info]
    if "cap" in df:
        capacity_info_str = df["cap"].tolist()
        capacity_value = [int(element) for element in capacity_info_str]
    else:
        capacity_value = weights*1000
    return edges, weights, capacity_value

def read_node_xy(node_xy_file_path):
    df = pd.read_csv(node_xy_file_path,index_col='id')
    df.index = df.index - 1
    return df

# create graph from edges and weights
def create_graph(edges, weights,capacity_value, debug):
    graph = ig.Graph(edges=edges, edge_attrs={"weight": weights}, directed=True)
    node_xy_df = read_node_xy(configs.PATH_CONFIG.RELATIVE_NODE_XY_PATH)
    for v in graph.vs:
        v["x"] = node_xy_df.loc[v.index]['x']
        v["y"] = node_xy_df.loc[v.index]['y']
        v["distance_to_hazard_after_scale"] = math.dist(constants.HAZARD,[v["x"],v["y"]])*configs.MAP_SCALE
        if debug:
            print(v["x"],v["y"])
    d_adj = np.zeros([graph.vcount(), graph.vcount()])
    for e, w in zip(edges, weights):
        d_adj[e[0], e[1]] = w
    if debug:
        print("d_adj: ", d_adj)
    cap = np.zeros([graph.vcount(), graph.vcount()])
    capacity_info = np.zeros([graph.vcount(), graph.vcount()])
    for e, c in zip(edges, capacity_value):
        cap[e[0], e[1]] = int(c)-configs.EXIST_FLOW
        capacity_info[e[0], e[1]] = int(c)
    if debug:
        print("cap: ", cap)
    return graph, d_adj, cap, capacity_info



# read OD pairs and demand from csv
def read_od_with_demand(od_raw, graph, debug, batch_size=None):
    df = pd.read_csv(od_raw, dtype=int, header=None)
    df_array = df.values.ravel()
    od_array = np.reshape(df_array, (len(df), 3))
    # (o, d, iteration)
    od_list = np.empty((0, 3), int)
    demand_list = np.empty((0, 1), int)
    node_xy_df = read_node_xy(configs.PATH_CONFIG.RELATIVE_NODE_XY_PATH)
    count_unconnected_od = 0
    for i in od_array:
        source_node = i[0]
        target_node = i[1]
        # print(node_xy_df.loc[target_node]['x'])
        # if (node_xy_df.loc[source_node]['y']> constants.SAFE_BOUNDRY) or (node_xy_df.loc[target_node]['y']<constants.SAFE_BOUNDRY):
        o_node_xy = [node_xy_df.loc[source_node]['x'],node_xy_df.loc[source_node]['y']]
        d_node_xy = [node_xy_df.loc[target_node]['x'],node_xy_df.loc[target_node]['y']]
        distance_from_haz = math.dist(constants.HAZARD,o_node_xy)
        distance_to_haz = math.dist(constants.HAZARD,d_node_xy)
        # if (node_xy_df.loc[source_node]['x']> constants.SAFE_BOUNDRY) or (node_xy_df.loc[target_node]['x']<constants.SAFE_BOUNDRY):
        #     continue
        if (distance_from_haz > constants.SAFE_BOUNDRY) or (distance_to_haz < constants.SAFE_BOUNDRY):
            continue
        else:
            shortest_path = graph.get_shortest_paths(
                source_node, to=target_node,weights=graph.es["weight"],output='vpath'
            )
            if shortest_path[0]==[]:
                print(f"No path from node {source_node} to node {target_node}")
                count_unconnected_od += 1
            else:
                _batch_size = i[-1] if batch_size is None else batch_size
                # _batch_size = 4 if batch_size is None else batch_size
                n_batch = i[-1] // _batch_size
                for batch in range(n_batch):
                    od_list = np.vstack([od_list, [i[0], i[1], batch]]) # Append the row to new_array
                    total_demand = _batch_size * configs.DEMAND_SCALE
                    o_node_xy = [node_xy_df.loc[source_node]['x'],node_xy_df.loc[source_node]['y']]
                    distance_from_haz = math.dist(constants.HAZARD,o_node_xy)
                    time_window = distance_from_haz/constants.HAZARD_SPEED
                    if time_window == 0:
                        flow_demand = total_demand
                    else:
                        # flow_demand = int(total_demand/time_window)
                        flow_demand = total_demand
                    demand_list = np.vstack([demand_list, int(flow_demand)])
    if debug:
        print("od_array", od_array)
    print("number of unconnected od pairs: ", count_unconnected_od)
    print("-"*50)
    return od_list, demand_list


def layer1():
    config = configs.LAYER1_CONFIGS
    debug = config.debug
    print('configs.PATH_CONFIG.RELATIVE_RAW_GRAPH_PATH:', configs.PATH_CONFIG.RELATIVE_RAW_GRAPH_PATH)
    print('configs.PATH_CONFIG.RELATIVE_OD_WITH_DEMAND_PATH:', configs.PATH_CONFIG.RELATIVE_OD_WITH_DEMAND_PATH)
    graph, d_adj,cap,capacity_info = create_graph(
        *read_graph(configs.PATH_CONFIG.RELATIVE_RAW_GRAPH_PATH), debug
    )
    od_pairs, demand_list = read_od_with_demand(configs.PATH_CONFIG.RELATIVE_OD_WITH_DEMAND_PATH, graph, debug, batch_size=configs.BATCH_SIZE)
    solvable_od = np.empty((0, 3), int)
    infessible_od = np.empty((0, 3), int)
    solvable_od_demand = np.empty((0, 1), int)
    infessible_od_demand = np.empty((0, 1), int)
    for od, demand in zip(od_pairs, demand_list):
        # Initlaize model
        model_params = ModelParams(
            graph=graph,
            od_pairs=[od],
            d_adj=d_adj,
            demand_list=[demand],
            layer=1,
            cap = cap,
            capacity_info = capacity_info,
        )
        model = init_model(model_params)
        # Formulate the optimization problem
        model = formulate_optimization_problem(
            model, layer=1, output_dir=config.result_dir
        )
        # Solves the model
        if solve_model(model, debug=debug,layer = 1):
            if debug:
                print('Layer1 solved')
            solvable_od = np.vstack([solvable_od, od])
            solvable_od_demand = np.vstack([solvable_od_demand, demand])
        else:
            print(f'not solved{od}')
            infessible_od = np.vstack([infessible_od, od])
            infessible_od_demand = np.vstack([infessible_od_demand, demand])
    if debug:
        print(f'solvable OD pairs:\n {solvable_od}')
        print(f'infeasible OD pairs:\n {infessible_od}')
    return solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap,capacity_info


def layer2(infessible_od, infessible_od_demand, d_adj, graph,cap,capacity_info):
    config = configs.LAYER2_CONFIGS
    debug = config.debug
    model_params = ModelParams(
            graph=graph,
            od_pairs=infessible_od,
            d_adj=d_adj,
            demand_list=infessible_od_demand,
            layer=2,
            cap = cap,
            capacity_info = capacity_info,
    )
    model = init_model(model_params)
    # Formulate the optimization problem
    model = formulate_optimization_problem(
        model, layer=2, output_dir=config.result_dir
    )
    # Solves the model
    if solve_model(model, debug=debug,layer=2):
        print('Layer2 solved')
        facilities = np.array([int(model.f[i].value) for i in model.nodes_set])
        if debug:
            print(facilities)
        if configs.SAVE_RESULTS:
            return save_results(model, config, debug, layer=2)
        return facilities
    else:
        return None


def layer3(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj, facilities, baseline,cap,capacity_info):
    config = configs.LAYER3_CONFIGS
    debug = config.debug
    # Initlaize model
    od_pairs = np.vstack((solvable_od, infessible_od))
    demand_list = np.vstack((solvable_od_demand, infessible_od_demand))
    model_params = ModelParams(
        graph=graph,
        od_pairs=od_pairs,
        d_adj=d_adj,
        demand_list=demand_list,
        facilities=facilities,
        layer=3,
        cap = cap,
        capacity_info = capacity_info,
    )
    model = init_model(model_params)
    # Formulate the optimization problem
    model = formulate_optimization_problem(model, layer=3, output_dir=config.result_dir)
    # Solves the model
    if solve_model(model, debug=debug,layer=3):
        print('Layer3 solved')
        if configs.SAVE_RESULTS:
            return save_results(model, config, debug, layer=3)
    else:
        print('Layer3 not solved')


def save_cost(model, origin, data):
    driving_time_cost = sum(
        model.d_adj[i, j] / model.s * model.x[i, j, k].value
        for i in model.nodes_set
        for j in model.nodes_set
        for k in model.OD_set
    )*configs.MAP_WEIGHT
    bpr_cost = sum(model.BPR_SLACK[i,j].value for i in model.nodes_set for j in model.nodes_set)*configs.BPR_WEIGHT
    # bpr_cost = (model.BPR_SLACK.value)*configs.BPR_WEIGHT
    charging_stop_cost = sum(model.y[i, k].value for i in model.nodes_set for k in model.OD_set)
    charging_time_cost = sum(model.e[i, k].value for i in model.nodes_set for k in model.OD_set)* model.t
    station_cost = sum(model.f[i].value for i in model.nodes_set)*configs.FACILITY_WEIGHT
    station_capacity_cost = sum(model.CS_SLACK[i].value for i in model.nodes_set)*configs.CS_WEIGHT
    # station_capacity_cost = model.CS_SLACK.value *configs.CS_WEIGHT
    cost_df = pd.DataFrame({
        'driving_time_cost': [driving_time_cost],
        'bpr_cost': [bpr_cost],
        'charging_stop_cost': [charging_stop_cost],
        'charging_time_cost': [charging_time_cost],
        'station_cost': [station_cost],
        'station_capacity_cost': [station_capacity_cost]
    }).transpose()
    data.cost[str(origin)] = cost_df
    return data


def od_layer3(od_pairs, demand_list, graph, d_adj, facilities, pre_charger_usage,cap, capacity_info,data=None):
    config = configs.LAYER3_CONFIGS
    debug = config.debug
    # Initlaize model
    model_params = ModelParams(
        graph=graph,
        od_pairs=od_pairs,
        d_adj=d_adj,
        demand_list=demand_list,
        facilities=facilities,
        layer=3,
        cap = cap,
        od_plans = data.od_plans if data is not None else {},
        pre_charger_usage=pre_charger_usage,
        capacity_info = capacity_info,
    )
    model = init_model(model_params)
    # Formulate the optimization problem
    model = formulate_optimization_problem(model, layer=3, output_dir=config.result_dir)
    # Solves the model
    if solve_model(model, debug=debug,layer=3) and configs.SAVE_RESULTS:
        print('Layer3 solved')
        if configs.SAVE_RESULTS:
            origin = od_pairs[0, 0]
            data = save_results(model, config, debug, layer=3, data=data)
            data = save_cost(model, origin, data)
            return data

def sort_od_to_hazard(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph):
    od_pairs = np.vstack((solvable_od, infessible_od))
    demand_list = np.vstack((solvable_od_demand, infessible_od_demand))
    # Get node names and their corresponding indices
    node_to_hazard = graph.vs["distance_to_hazard_after_scale"]
    node_indices = list(range(len(node_to_hazard)))
    # Sort node indices based on node names
    sorted_indices = sorted(node_indices, key=lambda i: node_to_hazard[i])
    # print(sorted_indices)
    dict_node_order = collections.OrderedDict()
    for node in sorted_indices:
        print(node)
        dict_node_order[str(node)] = []
    for idx,od in enumerate(od_pairs):
        demand = demand_list[idx][0]
        o = str(od[0])
        row = [od[0],od[1],od[2],demand]
        dict_node_order[o].append(row)
    return dict_node_order



def combined_layer4(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap,capacity_info):
    config = configs.LAYER4_CONFIGS
    debug = config.debug
    od_pairs = np.vstack((solvable_od, infessible_od))
    demand_list = np.vstack((solvable_od_demand, infessible_od_demand))
    model_params = ModelParams(
            graph=graph,
            od_pairs=od_pairs,
            d_adj=d_adj,
            demand_list=demand_list,
            layer=4,
            cap = cap,
            capacity_info = capacity_info,
    )
    model = init_model(model_params)
    # Formulate the optimization problem
    model = formulate_optimization_problem(
        model, layer=4, output_dir=config.result_dir
    )
    # Solves the model
    if solve_model(model, debug=debug,layer=4):
        print('Layer4 solved')
        facilities = np.array([int(model.f[i].value) for i in model.nodes_set])
        if debug:
            print(facilities)
        if configs.SAVE_RESULTS:
            return save_results(model, config, debug, layer=4)


def save_results(model, config, debug, layer, data=None):
    if data is None:
        data = RunData()
    save_flow_size(model, data, debug, layer)
    data = save_y_e_demand(model,data,layer)
    data = save_average_wait_time(model,data,debug,layer)
    df_result(model, data, debug, layer)
    save_charging_demand_size(model, data, debug, layer)
    data.model = model
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    output_excel_path = os.path.join(config.result_dir, config.output_excel_name)
    with pd.ExcelWriter(output_excel_path) as writer:
        data.metadata.to_excel(writer, sheet_name="metadata", index=False)
        data.flow_size.to_excel(writer, sheet_name="flow_size", index=False)
        data.actual_od_flow.to_excel(writer, sheet_name="actual_od_flow", index=False)
        data.demanded_od_flow.to_excel(writer, sheet_name="demanded_od_flow", index=False)
        data.evacuation_time.to_excel(writer, sheet_name="evacuation_time", index=False)
        data.charging_demand_size.to_excel(writer, sheet_name="charging_demand_size", index=True)
        for od_pair, od_plan in data.od_plans.items():
            od_plan.to_excel(writer, sheet_name="-".join([str(node) for node in od_pair]), index=False)
    return data


def combined_layer5(solvable_od, solvable_od_demand, infessible_od, infessible_od_demand, graph, d_adj,cap):
    config = configs.LAYER5_CONFIGS
    debug = config.debug
    od_pairs = np.vstack((solvable_od, infessible_od))
    demand_list = np.vstack((solvable_od_demand, infessible_od_demand))
    model_params = ModelParams(
            graph=graph,
            od_pairs=od_pairs,
            d_adj=d_adj,
            demand_list=demand_list,
            layer=5,
            cap = cap,
    )
    model = init_model(model_params)
    # Formulate the optimization problem
    model = formulate_optimization_problem(
        model, layer=5, output_dir=config.result_dir
    )
    # Solves the model
    if solve_model(model, debug=debug,layer=5):
        print('Layer5 solved')
        facilities = np.array([int(model.f[i].value) for i in model.nodes_set])
        if debug:
            print(facilities)
        if configs.SAVE_RESULTS:
            return save_results(model, config, debug, layer=5)


@dataclasses.dataclass
class BaselineOutput:
    no_charging_od: np.ndarray
    no_charging_od_demand: np.ndarray
    no_charging_od_route: dict
    charging_od: np.ndarray
    charging_od_demand: np.ndarray
    charging_od_route: dict
    infeasible_od: np.ndarray
    infeasible_od_demand: np.ndarray
    graph: ig.Graph
    d_adj: np.ndarray
    cap: np.ndarray
    capacity_info: np.ndarray
    charging_od_solution: list
    nodes_size: int
    nodes_set: range
    OD_set: range
    f: np.ndarray
    o: np.ndarray
    demand_list: np.ndarray
    x: np.ndarray
    OD: np.ndarray
    s: int
    e: np.ndarray
    t: int
    r: np.ndarray
    y: np.ndarray
    max_road_flow: np.ndarray

def make_random_station(nodes_size, number_to_make):
    station = []
    rand_station = random_existing_stations(0, nodes_size - 1, number_to_make)
    station.extend(rand_station)
    return station

def baseline():
    config = configs.BASELINE_CONFIGS
    debug = config.debug
    print('configs.PATH_CONFIG.RELATIVE_RAW_GRAPH_PATH:', configs.PATH_CONFIG.RELATIVE_RAW_GRAPH_PATH)
    print('configs.PATH_CONFIG.RELATIVE_OD_WITH_DEMAND_PATH:', configs.PATH_CONFIG.RELATIVE_OD_WITH_DEMAND_PATH)
    graph, d_adj,cap,capacity_info = create_graph(*read_graph(configs.PATH_CONFIG.RELATIVE_RAW_GRAPH_PATH), debug)
    od_pairs, demand_list = read_od_with_demand(configs.PATH_CONFIG.RELATIVE_OD_WITH_DEMAND_PATH, graph, debug)
    nodes_size = graph.vcount()
    #randomly select existing stations
    # exist_station = make_random_station(nodes_size, configs.COUNT_EXISTING_STATIONS)
    exist_station = configs.PRESET_STATION
    print(f"Add existing station at node {exist_station}")
    init_fuel=round(constants.VEHICLE_DRIVING_RANGE * constants.FUEL_MULTIPLIER)
    model = BaselineOutput(
        no_charging_od=np.empty((0, 3), int),
        no_charging_od_demand=np.empty((0, 1), int),
        no_charging_od_route = {},
        charging_od=np.empty((0, 3), int),
        charging_od_demand=np.empty((0, 1), int),
        charging_od_route = {},
        infeasible_od=np.empty((0, 3), int),
        infeasible_od_demand=np.empty((0, 1), int),
        graph=graph,
        d_adj=d_adj,
        cap=cap,
        capacity_info = capacity_info,
        charging_od_solution=[],
        nodes_size = nodes_size,
        nodes_set = range(nodes_size),
        OD_set = [],
        f = np.zeros(nodes_size),
        o = np.zeros(nodes_size),
        demand_list = demand_list,
        x = np.zeros([nodes_size, nodes_size, len(od_pairs)]),
        OD = np.empty((0, 2), int),
        s = constants.DRIVING_SPEED,
        e = np.zeros([nodes_size, len(od_pairs)]),
        t = constants.RECHARGE_TIME,
        r = np.zeros([nodes_size, len(od_pairs)]),
        y = np.zeros([nodes_size, len(od_pairs)]),
        max_road_flow=np.zeros([nodes_size, nodes_size]),
    )
    for i in exist_station:
        model.o[i] = 1
    model.f = model.o
    time_start = time.time()
    for od, demand in zip(od_pairs, demand_list):
        # calculate the shortest distance for the OD pair
        shortest_path = graph.get_shortest_paths(od[0], to=od[1],weights=graph.es["weight"],output="epath")
        distance = 0
        path =[]
        for e in shortest_path[0]:
            distance += graph.es[e]["weight"]
            path.append([graph.es[e].source,graph.es[e].target])
        # add ODs can reach destination without charging to the no_charging_od list
        if distance <= init_fuel:
            model.no_charging_od = np.vstack([model.no_charging_od, od])
            model.no_charging_od_demand = np.vstack([model.no_charging_od_demand, demand])
            model.no_charging_od_route[tuple(od)] = path
        else:
            charging_solution_set = []
            charging_solution_set_path = {}
            for i in exist_station:
                o_to_charger = graph.get_shortest_paths(od[0], to=i,weights=graph.es["weight"],output="epath")
                o_to_charger_distance = 0
                o_to_charger_path = []
                for e in o_to_charger[0]:
                    o_to_charger_distance += graph.es[e]["weight"]
                    o_to_charger_path.append([graph.es[e].source,graph.es[e].target])
                # Identify the charging stations that can be reached from the origin.
                if (o_to_charger[0]!=[]) and (o_to_charger_distance<= init_fuel):
                    # Check if the destination node is within "reach" of this charging station.
                    charger_to_d = graph.get_shortest_paths(i, to=od[1],weights=graph.es["weight"],output="epath")
                    charger_to_d_distance = 0
                    charger_to_d_path = []
                    for e in charger_to_d[0]:
                        charger_to_d_distance += graph.es[e]["weight"]
                        charger_to_d_path.append([graph.es[e].source,graph.es[e].target])
                    if (charger_to_d[0]!=[]) and (charger_to_d_distance <= constants.VEHICLE_DRIVING_RANGE):
                        charged_range = charger_to_d_distance + o_to_charger_distance - init_fuel

                        # if yes, add the path to the candidate solution set
                        total_distance = o_to_charger_distance + charger_to_d_distance
                        driving_time = total_distance/constants.DRIVING_SPEED/60
                        charging_time = (charged_range*constants.RECHARGE_TIME)
                        total_time = driving_time + charging_time
                        charging_solution_set.append([i,total_distance])
                        charging_solution_set_path[i] = {"o_to_charger_path": o_to_charger_path,"charger_to_d_path": charger_to_d_path,"charging station":i,"charged_range": charged_range,"total_distance":total_distance,"driving_time":driving_time,"charging_time":charging_time,"total_time":total_time}
            # if no solution is found, add the OD to the infeasible_od list
            if len(charging_solution_set) == 0:
                model.infeasible_od = np.vstack([model.infeasible_od, od])
                model.infeasible_od_demand = np.vstack([model.infeasible_od_demand, demand])
            # if multiple solutions are found, select the shortest one
            else:
                shortest_solution = min(charging_solution_set,key=lambda x:x[1])
                model.charging_od_solution.append(shortest_solution)
                model.charging_od = np.vstack([model.charging_od, od])
                model.charging_od_demand = np.vstack([model.charging_od_demand, demand])
                model.charging_od_route[tuple(od)] = charging_solution_set_path[shortest_solution[0]]
    time_end = time.time()
    time_cost = time_end - time_start

    model.OD = np.vstack((model.no_charging_od,model.charging_od))
    model.n = np.vstack((model.no_charging_od_demand,model.charging_od_demand))
    model.demand_list = np.vstack((model.no_charging_od_demand,model.charging_od_demand))
    model.OD_set = range(len(model.OD))
    for k,od in enumerate(model.OD):
        if tuple(od) in (tuple(row) for row in model.no_charging_od):
            model.e[:,k] = 0
            model.y[:,k] = 0
            path = model.no_charging_od_route[tuple(od)]
            for i in path:
                from_node = i[0]
                to_node = i[1]
                model.x[from_node,to_node,k] = 1
                if from_node == od[0]:
                    model.r[from_node,k] = init_fuel
                model.r[to_node,k] = model.r[from_node,k] - model.d_adj[from_node,to_node]
        else:
            charging_place = model.charging_od_route[tuple(od)]["charging station"]
            charging_amount = model.charging_od_route[tuple(od)]["charged_range"]
            model.e[charging_place,k] = charging_amount
            model.y[charging_place,k] = 1
            path_o_to_charger = model.charging_od_route[tuple(od)]["o_to_charger_path"]
            for i in path_o_to_charger:
                from_node = i[0]
                to_node = i[1]
                model.x[from_node,to_node,k] = 1
                if from_node == od[0]:
                    model.r[from_node,k] = init_fuel
                model.r[to_node,k] = model.r[from_node,k] - model.d_adj[from_node,to_node]

            path_charger_to_d = model.charging_od_route[tuple(od)]["charger_to_d_path"]
            for i in path_charger_to_d:
                from_node = i[0]
                to_node = i[1]
                model.x[from_node,to_node,k] = 1
                if from_node == charging_place:
                    model.r[to_node,k] = model.r[from_node,k] - model.d_adj[from_node,to_node]+charging_amount
                else:
                    model.r[to_node,k] = model.r[from_node,k] - model.d_adj[from_node,to_node]
                model.x[i[0],i[1],k] = 1

    for i in model.nodes_set:
        for j in model.nodes_set:
            for k in model.OD_set:
                if model.x[i,j,k] == 1:
                    model.max_road_flow[i, j] = model.max_road_flow[i, j] + model.n[k]


    if configs.SAVE_RESULTS:
        data = save_results(model, config, debug, layer=0)
        baseline_time_path = os.path.join(config.result_dir, "baseline_time.txt")
        with open(baseline_time_path, "w") as f:
            print(f"Time cost: {time_cost}",file=f)
            print("-"*50,file=f)
            for k in model.OD_set:
                if k in model.no_charging_od_route:
                    print(f"no charging OD: {model.no_charging_od_route[k]}",file=f)
                if k in model.charging_od_route:
                    print(f"charging OD: {model.charging_od_route[k]}",file=f)
            print("-"*50,file=f)
            print(f"no_charging_od:\n {model.no_charging_od}",file=f)
            print("total number of ODs require no charging: ", len(model.no_charging_od),file=f)
            print("-"*50,file=f)
            print(f"charging_od:\n {model.charging_od}",file=f)
            print(f"charging_od_solution:([charging station, total travel time])\n {model.charging_od_solution}",file=f)
            print("number of ODs require charging: ", len(model.charging_od),file=f)
            print("-"*50,file=f)
            print(f"infeasible_od:\n {model.infeasible_od}",file=f)
            print("number of ODs infeasible: ", len(model.infeasible_od),file=f)
    return model, data