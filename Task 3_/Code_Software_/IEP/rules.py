import configs
import constants
import pyomo.environ as pyo
import math
from pynverse import inversefunc


#helper functions
def calculate_wait_time(rho):
    c = configs.CHARGER_PROT_NUM
    c_factorial= (math.factorial(c))
    max_charge_time = configs.MAX_CHARGING_TIME
    mu = 1/max_charge_time
    lamb = rho * c * mu
    r = lamb/mu
    term1 = (r**c)/(c_factorial*(1-rho))

    term2 = 0
    for n in range(c):
        term2 = term2+((r**n)/(math.factorial(n)))
    term3 = (term1)/(c*mu*(1-rho))
    p0 = 1/(term1 + term2)
    t_wait = (term3)*(p0)
    return t_wait

def calculate_max_rho():
    max_wait_time = configs.MAX_WAIT_TIME
    invf = inversefunc(calculate_wait_time,domain=[0, 0.9999999999999999])
    return(invf(max_wait_time))

# cost function for layer 1
def layer1_cost_(model):
    return (
        sum(
            model.d_adj[i, j] * model.x[i, j, k]
            for i in model.nodes_set
            for j in model.nodes_set
            for k in model.OD_set
        )
    )


# cost function for layer 2 and 3
def layer2_cost_(model):
    return (
        sum(
            model.d_adj[i, j] / configs.MAP_SCALE/model.s * model.x[i, j, k]
            for i in model.nodes_set
            for j in model.nodes_set
            for k in model.OD_set
        )
        + sum(model.y[i, k] for i in model.nodes_set for k in model.OD_set)
        + sum(model.f[i] for i in model.nodes_set)
        + sum(model.e[i, k] for i in model.nodes_set for k in model.OD_set) * model.t
        #try
        #+ sum(10000*model.BPR_SLACK[i,j] for i in model.nodes_set for j in model.nodes_set)
    )

def layer3_cost_(model):
    return (
        (sum(
            model.d_adj[i, j] / model.s * model.x[i, j, k]
            for i in model.nodes_set
            for j in model.nodes_set
            for k in model.OD_set
        )*configs.MAP_WEIGHT
        + sum(model.BPR_SLACK[i,j] for i in model.nodes_set for j in model.nodes_set)*configs.BPR_WEIGHT*len(model.OD)
        #  + sum(model.capacity_info[i,j]*model.BPR_SLACK[i,j] for i in model.nodes_set for j in model.nodes_set)*configs.BPR_WEIGHT
        # + sum(model.y[i, k] for i in model.nodes_set for k in model.OD_set)
        # + model.BPR_SLACK * configs.BPR_WEIGHT
        + sum(model.e[i, k] for i in model.nodes_set for k in model.OD_set)* model.t)/len(model.OD)
        + sum(model.f[i] for i in model.nodes_set)*configs.FACILITY_WEIGHT
        # + model.CS_SLACK*configs.CS_WEIGHT
         + sum(model.CS_SLACK_OD[i,k] for i in model.nodes_set for k in model.OD_set)*configs.CS_WEIGHT
        # + sum((model.demand_list[k]-model.n[k])**2 for k in model.OD_set)*configs.DEMAND_WEIGHT
    )

def layer4_cost_(model):
    return (
        (sum(
            model.d_adj[i, j] / model.s * model.x[i, j, k]
            for i in model.nodes_set
            for j in model.nodes_set
            for k in model.OD_set
        ) * configs.MAP_WEIGHT
        + sum(model.BPR_SLACK[i,j] for i in model.nodes_set for j in model.nodes_set)*configs.BPR_WEIGHT*len(model.OD)
        # + sum(model.capacity_info[i,j]*model.BPR_SLACK[i,j] for i in model.nodes_set for j in model.nodes_set)*configs.BPR_WEIGHT
        # + model.BPR_SLACK * configs.BPR_WEIGHT
        # + sum(model.y[i, k] for i in model.nodes_set for k in model.OD_set)
        + sum(model.e[i, k] for i in model.nodes_set for k in model.OD_set) * model.t)/len(model.OD)
        + sum(model.f[i] for i in model.nodes_set)*configs.FACILITY_WEIGHT
        # + model.CS_SLACK*configs.CS_WEIGHT
        + sum(model.CS_SLACK_OD[i,k] for i in model.nodes_set for k in model.OD_set)*configs.CS_WEIGHT
    )


def layer5_cost_(model):
    return (
        sum(
            model.d_adj[i, j] / model.s * model.x[i, j, k]
            # model.d_adj[i, j] / model.s * model.x[i, j, k]*model.demand_list[k]
            for i in model.nodes_set
            for j in model.nodes_set
            for k in model.OD_set
        )*configs.MAP_WEIGHT
        + sum(model.BPR_SLACK[i,j,k] for i in model.nodes_set for j in model.nodes_set)*configs.BPR_WEIGHT
        + sum(model.y[i, k]*model.demand_list[k]  for i in model.nodes_set for k in model.OD_set)
        + sum(model.e[i, k]*model.demand_list[k]  for i in model.nodes_set for k in model.OD_set) * model.t
        + sum(model.f[i] for i in model.nodes_set) + sum((model.demand_list[k]-model.n[k])**2 for k in model.OD_set)
    )

def layer3_cost_baseline_(model):
    return (
        sum(
            model.d_adj[i, j] / model.s * model.x[i, j, k]
            for i in model.nodes_set
            for j in model.nodes_set
            for k in model.OD_set
        )
        + sum(model.y[i, k] for i in model.nodes_set for k in model.OD_set)
        + sum(model.e[i, k] for i in model.nodes_set for k in model.OD_set) * model.t
    )


def BPR_fun_(model,i,j,flow):
    """Return BPR function value.

    Flow is the solution solved from the optimization model a function of model, i & j.
    """
    return 1 + 0.15 * (flow / model.cap[i,j]) ** 4


def charger_usage_const_layer3_(model, i):
    return sum(model.y[i, k]*model.demand_list[k] for k in model.OD_set) == model.charger_usage[i]
    # return sum(model.y[i, k]*model.demand_list[k] for k in model.OD_set) == model.charger_usage[i]+model.pre_charger_usage[i]

def charger_usage_const_layer4_(model, i):
    return sum(model.y[i, k]*model.demand_list[k] for k in model.OD_set) == model.charger_usage[i]
    # return sum(model.y[i, k]*model.demand_list[k] for k in model.OD_set) == model.charger_usage[i]+model.pre_charger_usage[i]


def charger_usage_const_layer5_(model, i):
    return sum(model.e[i, k]*model.n[k] for k in model.OD_set) == model.charger_usage[i]

def max_charger_usage_layer2_(model, i):
    if model.o[i]==0:
        return model.charger_usage[i] <= (configs.MOBILE_CHARGER_CAPACITY)
    elif model.o[i]==1:
        return model.charger_usage[i] <= configs.FIXED_CHARGER_CAPACITY
    # else:
    #     return pyo.Constraint.Skip


def max_charger_usage_layer3_(model, i):
    return model.charger_usage[i]+model.pre_charger_usage[i] <= calculate_max_rho()*configs.CHARGER_PROT_NUM/configs.MAX_CHARGING_TIME+model.CS_SLACK[i]

def max_charger_usage_layer4_(model, i):
    return model.charger_usage[i] <= calculate_max_rho()*configs.CHARGER_PROT_NUM/configs.MAX_CHARGING_TIME+model.CS_SLACK[i]


# def max_charger_usage_layer3_(model, i):
#     if (model.o[i]==0):
#         return model.charger_usage[i]+model.pre_charger_usage[i] <= (configs.MOBILE_CHARGER_CAPACITY)+model.CS_SLACK[i]
#         # return model.charger_usage[i]+model.pre_charger_usage[i] <= (configs.MOBILE_CHARGER_CAPACITY)+model.CS_SLACK
#     elif model.o[i]==1:
#         # return model.charger_usage[i] +model.pre_charger_usage[i]<= (configs.FIXED_CHARGER_CAPACITY)+model.CS_SLACK
#         return model.charger_usage[i] +model.pre_charger_usage[i]<= (configs.FIXED_CHARGER_CAPACITY)+model.CS_SLACK[i]
#     else:
#         return pyo.Constraint.Skip
# def max_charger_usage_layer4_(model, i):
#     if (model.o[i]==0):
#         return model.charger_usage[i]<= (configs.MOBILE_CHARGER_CAPACITY)+model.CS_SLACK[i]
#     elif model.o[i]==1:
#         return model.charger_usage[i] <= (configs.FIXED_CHARGER_CAPACITY)+model.CS_SLACK[i]
#     else:
#         return pyo.Constraint.Skip

# calculates max road usage
def max_road_flow_constraint_(model, i, j):
    return (
        sum(model.x[i, j, k] * model.demand_list[k] for k in model.OD_set) == model.max_road_flow[i, j]
    )

def max_road_flow_constraint_layer5_(model, i, j):
    return (
        sum(model.x[i, j, k] * model.n[k] for k in model.OD_set) == model.max_road_flow[i, j]
    )


# only charge at placed station
def layer12_facility_(model, i, k):
    return model.y[i, k] <= model.f[i]

def layer3_facility_(model, i, k):
    return model.y[i, k] <= model.f[i]
    # if model.f[i] == 0:
    #     return model.y[i, k] == 0
    # else:
    #     return pyo.Constraint.Skip


# only adding new facility
def facility_sum_(model):
    return sum(model.f[k] for k in model.nodes_set) - sum(model.o) == configs.MAX_CHARGER_NUMBER


# only adding new facility
def facility_sum_max_(model):
    return sum(model.f[k] for k in model.nodes_set) - sum(model.o) <= configs.MAX_CHARGER_NUMBER


# flow constraint
def flow_(model, i, k):
    if i == model.OD[k][0]:
        return sum([model.x[i, j, k] for j in model.nodes_set]) - sum([model.x[j, i, k] for j in model.nodes_set]) == 1
    elif i == model.OD[k][1]:
        return sum([model.x[j, i, k] for j in model.nodes_set]) -sum([model.x[i, j, k] for j in model.nodes_set]) == 1
    else:
        return (
        sum([model.x[i, j, k] for j in model.nodes_set])
        - sum([model.x[j, i, k] for j in model.nodes_set])
        == 0
        )


# only passing existing edges
def openedge_(model, i, j, k):
    if model.d_adj[i, j] == 0:
        return model.x[i, j, k] == 0
    else:
        return pyo.Constraint.Skip


# foreard path capacity lower constraint
def forward_lower_(model, i, j, k):
    return (
        model.r[i, k]
        + model.e[i, k]
        - model.d_adj[i, j]
        - model.B * (1 - model.x[i, j, k])
        <= model.r[j, k]
    )


# foreard path capacity upper constraint
def forward_upper_(model, i, j, k):
    return (
        model.r[i, k]
        + model.e[i, k]
        - model.d_adj[i, j]
        + model.B * (1 - model.x[i, j, k])
        >= model.r[j, k]
    )


# determine vehicle recharge or not
def refuel_decision_(model, i, k):
    return model.e[i, k] == model.V * model.y[i, k] - model.a[i, k]


# determine vehicle refuel level
def refuel_constraint_(model, i, k):
    return model.e[i, k] <= model.V - model.r[i, k]


# init fuel level
def init_fuel_(model, k):
    O = model.OD[k][0]
    # return model.r[O,k]==round(model.V*random.uniform(0, 1))
    return model.r[O, k] == round(model.V * constants.FUEL_MULTIPLIER)


# only recharge on the route
def can_refuel_(model, i, k):
    return model.y[i, k] <= sum(model.x[i, j, k] for j in model.nodes_set)


# exist charging station
def exist_facility_(model, i):
    return model.f[i] >= model.pre_f[i]

def BPR_constraint_(model, i, j):
    if model.cap[i,j] == 0:
        return pyo.Constraint.Skip
    else:
    #     return model.max_road_flow[i,j]/model.cap[i,j]<= 1 + model.BPR_SLACK[i,j]
        return model.max_road_flow[i, j]/model.cap[i,j] <= 1 + model.BPR_SLACK[i,j]
    # return model.max_road_flow[i, j] <= configs.ROAD_CAPACITY+model.BPR_SLACK[i, j]

def BPR_OD_constraint_(model, i, j):
    return model.max_road_flow[i, j] <= model.cap[i,j]+model.BPR_SLACK[i,j]

def noloop_(model, i, k):
    return sum([model.x[i, j, k] for j in model.nodes_set]) <= 1

def big_M_CS_constraint_upper_(model, i, k):
    return model.CS_SLACK_OD[i,k] <= model.CS_SLACK[i]+model.B*(1-model.y[i,k])

def big_M_CS_constraint_lower_(model, i, k):
    return model.CS_SLACK_OD[i,k] >= model.CS_SLACK[i]-model.B*(1-model.y[i,k])

def fix_od_constraint_(model, i, j, k):
    if (len(model.od_plans.keys())>0) and (k < len(model.od_plans.items())):
        key = list(model.od_plans.keys())[k]
        value_frame = model.od_plans[key]
        route_frame = list(zip(value_frame['i'], value_frame['j']))
        if (i,j) in route_frame:
            return model.x[i, j, k] == 1
        else:
            return model.x[i, j, k] == 0
    else:
        return pyo.Constraint.Skip
