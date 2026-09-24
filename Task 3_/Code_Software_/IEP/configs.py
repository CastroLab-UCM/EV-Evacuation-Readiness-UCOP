"""Configs to run the graph for the layer model"""
import dataclasses
from datetime import datetime
import os
import platform

################################
## Debugging Parameters
################################
# Available mode: 'gui' or 'non-gui'. Effectively anything not 'gui' will trigger non-gui mode.
# MODE = 'non-gui'
MODE = 'gui'
LAYER = 'BASELINE'
# LAYER = 'OPT'
# LAYER = 'SINGLE'
FLOW_EDGE = True

MAP_SCALE =1
MAX_CHARGER_NUMBER = 1
CHARGER_PROT_NUM = 5
MOBILE_CHARGER_CAPACITY = 5*38*CHARGER_PROT_NUM

FIXED_CHARGER_CAPACITY = MOBILE_CHARGER_CAPACITY
FLOW_SIMULATION_TIME = 1
SIMULATION_FLOW_END_TIME = str(3600*FLOW_SIMULATION_TIME)
EXIST_FLOW = 0
MAX_WAIT_TIME = 0.5
MAX_CHARGING_TIME = 0.2
DEMAND_SCALE = 0.001*50
################################
## Discrete Scannable Parameters
################################
# MAP_WEIGHT_TUNE_VALUES = [1000, 100, 10, 1]
MAP_WEIGHT_TUNE_VALUES = [1]

# if MAP_WEIGHT_TUNE_VALUES is set, MAP_WEIGHT will be overwritten in m:
MAP_WEIGHT = 1
BPR_WEIGHT = 1
DEMAND_WEIGHT = 1
# CS_WEIGHT = MAX_WAIT_TIME
CS_WEIGHT = 1
FACILITY_WEIGHT = 10
# FACILITY_WEIGHT = 10
#############################
## Shared Configs
#############################
#no charging station
# sol = []
#SiouxFalls charging station
# sol = [5]
# sol = [15,41,44]
sol = [42,46]
#FL CHARGING STATION
# sol=[3,4,5,10,12,13,14,15,16,17,18,20,21,22,43,44,45,46,48,77,75,74,73,67,40,59,65,63,62]
PRESET_STATION = [ele -1 for ele in sol]

SAVE_RESULTS = True
FACILITY_SUM_CONSTRAINT = True

BATCH_SIZE = None

#######################
## Path configs
#######################
_TIMENOW = datetime.now().strftime("%Y%m%d_%H%M")
_MAC_PYTHON_CODES_DIR = '/Users/shuangfeng/Documents/git/P_24_Evacuation_Plan/python_codes'
_WINDOWS_PYTHON_CODES_DIR = 'C:\\Users\\Shuang\\Documents\\GitHub\\P_24_Evacuation_Plan\\python_codes'
_MAC_DATA_DIR = '/Users/shuangfeng/Library/CloudStorage/Box-Box/evacuation_data/'
_WINDOWS_DATA_DIR = 'C:\\Users\\Shuang\\Box\\evacuation_data'
# /Users/shuangfeng/Documents/git/P_24_Evacuation_Plan/python_codes/SiouxFalls_oneway_customOD

if platform.system() == 'Darwin':
    PYTHON_CODES_DIR = _MAC_PYTHON_CODES_DIR
    DATA_DIR=_MAC_DATA_DIR
elif platform.system() == 'Windows':
    PYTHON_CODES_DIR = _WINDOWS_PYTHON_CODES_DIR
    DATA_DIR=_WINDOWS_DATA_DIR
else:
    raise Exception(f'Unsupported platform: {platform.system()}')
PLATFORM = platform.system()
BASE_DIR = os.path.join(DATA_DIR, 'results', _TIMENOW)
_AVAILABLE_CASES = [
    'N_D',
    'SiouxFalls',
    'FL',
    'MP',
    'Dummy',
    'SiouxFalls_oneway',
]
# Choose one of the cases above. Ultimately we want to randomly generate the graph
# CASE_NAME = 'Mariposa'
# CASE_NAME = 'Dummy'
CASE_NAME = 'mariposa_small'


class PathConfig:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.data_dir = DATA_DIR
        self.python_codes_dir = PYTHON_CODES_DIR

    @property
    def BASE_DIR(self):
        return self.base_dir

    @property
    def DATA_DIR(self):
        return self.data_dir

    @property
    def PYTHON_CODES_DIR(self):
        return self.python_codes_dir

    @property
    def RELATIVE_RAW_GRAPH_PATH(self):
        return os.path.join(self.python_codes_dir, CASE_NAME, f'{CASE_NAME}.csv')

    @property
    def RELATIVE_OD_WITH_DEMAND_PATH(self):
        return os.path.join(self.python_codes_dir, CASE_NAME, f'{CASE_NAME}_od_demand.csv')

    @property
    def RELATIVE_NODE_XY_PATH(self):
        return os.path.join(self.python_codes_dir, CASE_NAME, f'{CASE_NAME}_xy.csv')

    @property
    def RELATIVE_NODE_XML_PATH(self):
        return os.path.join(self.python_codes_dir, CASE_NAME, f'{CASE_NAME}.nodes.xml')

    @property
    def RELATIVE_EDGE_XML_PATH(self):
        return os.path.join(self.python_codes_dir, CASE_NAME, f'{CASE_NAME}.edges.xml')

PATH_CONFIG = PathConfig()

RELATIVE_RAW_GRAPH_PATH = os.path.join(PYTHON_CODES_DIR, CASE_NAME, f'{CASE_NAME}.csv')
RELATIVE_OD_WITH_DEMAND_PATH = os.path.join(
    PYTHON_CODES_DIR, CASE_NAME, f'{CASE_NAME}_od_demand.csv'
)
RELATIVE_NODE_XY_PATH = os.path.join(
    PYTHON_CODES_DIR, CASE_NAME, f'{CASE_NAME}_xy.csv'
)
RELATIVE_NODE_XML_PATH = os.path.join(PYTHON_CODES_DIR, CASE_NAME, f'{CASE_NAME}.nodes.xml')
RELATIVE_EDGE_XML_PATH = os.path.join(PYTHON_CODES_DIR, CASE_NAME, f'{CASE_NAME}.edges.xml')


#############################
## Layer Configs
#############################

@dataclasses.dataclass
class _LayerConfigs():

    """Configs for each layer"""
    debug: bool = False
    save_result: bool = False
    result_dir_suffix: str | None = None
    output_excel_name: str | None = None
    output_excel_name2: str | None = None
    output_model_constraints: list[str] | None = None
    cost_output_excel_name: str | None = None

    @property
    def result_dir(self):
        return os.path.join(PATH_CONFIG.BASE_DIR, f'{CASE_NAME}_{self.result_dir_suffix}')

BASELINE_CONFIGS=_LayerConfigs(
    debug=False,
    save_result=True,
    result_dir_suffix='baseline',
    output_excel_name=f'{_TIMENOW}_baseline.xlsx',
    output_excel_name2 = f'{_TIMENOW}_baseline_sorted.xlsx',
    output_model_constraints=[
    ]
)

LAYER1_CONFIGS = _LayerConfigs(
    debug=False,
    result_dir_suffix='layer1',
    output_excel_name=f'{_TIMENOW}_layer1.xlsx',
)

LAYER2_CONFIGS = _LayerConfigs(
    debug=False,
    save_result=True,
    result_dir_suffix='layer2',
    output_excel_name=f'{_TIMENOW}_layer2.xlsx',
    output_model_constraints=[
        'openedge',
        'flow',
    ]
)

LAYER3_CONFIGS = _LayerConfigs(
    debug=False,
    save_result=True,
    result_dir_suffix='layer3',
    output_excel_name=f'{_TIMENOW}_layer3.xlsx',
    cost_output_excel_name=f'{_TIMENOW}_cost.xlsx',
    output_excel_name2 = f'{_TIMENOW}_layer3_sorted.xlsx',
    output_model_constraints=[
        #'facility',
        'openedge',
        'flow',
        'forward_lower',
        'forward_upper',
        'refuel_decision',
        'refuel_constraint',
        'init_fuel',
        'can_refuel',
        # 'max_charger_usage',
        # 'charger_usage_const',
    ]
)

LAYER4_CONFIGS = _LayerConfigs(
    debug=False,
    save_result=True,
    result_dir_suffix='layer4',
    output_excel_name=f'{_TIMENOW}_layer4.xlsx',
    output_excel_name2 = f'{_TIMENOW}_layer4_sorted.xlsx',
    output_model_constraints=[
        # 'facility',
        # 'openedge',
        # 'flow',
        # 'forward_lower',
        # 'forward_upper',
        # 'refuel_decision',
        # 'refuel_constraint',
        # 'init_fuel',
        # 'can_refuel',
        # 'max_charger_usage',
        # 'charger_usage_const',
    ]
)

LAYER5_CONFIGS = _LayerConfigs(
    debug=False,
    save_result=True,
    result_dir_suffix='layer5',
    output_excel_name=f'{_TIMENOW}_layer5.xlsx',
    output_model_constraints=[
        # 'facility',
        # 'openedge',
        # 'flow',
        # 'forward_lower',
        # 'forward_upper',
        # 'refuel_decision',
        # 'refuel_constraint',
        # 'init_fuel',
        # 'can_refuel',
        # 'max_charger_usage',
        # 'charger_usage_const',
    ]
)

LAYER_CONFIGS = {
    0: BASELINE_CONFIGS,
    1: LAYER1_CONFIGS,
    2: LAYER2_CONFIGS,
    3: LAYER3_CONFIGS,
    4: LAYER4_CONFIGS,
    5: LAYER5_CONFIGS,
}

#######################
## Unused configs
#######################

# number of breakpoints for bpr function
N_BREAKPOINTS = 1
BASELINE = False
FIGURE_DPI = 100
