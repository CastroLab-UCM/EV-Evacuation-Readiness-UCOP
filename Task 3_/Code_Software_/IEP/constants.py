import configs
"""Constants used in the layer model scan params script."""

#########################################################
## Optimization Constants
#########################################################
VERY_LARGE_NUMBER = 1e9

# Vehicle driving range in mile/fullbattery
VEHICLE_DRIVING_RANGE = 243

# 64kwh/10kw=6.4 hour for full range charging
# 6.4h charging time for 243 mile range
#38 mile/hour for L2 charger as 1/38 hour/mile

# 64kwh/50kw=1.28 hour for full range charging
# 1.28h charging time for 243 mile range
#190 mile/hour for fastcharging charger as 1/190 hour/mile

RECHARGE_TIME = 1 / 190
# 25 mile/hour for L2 charger
#RECHARGE_TIME = 1 / 25 / 60

#64 mile/hour = 1.07 mile/min
# Driving speed in miles/min
# highway
# DRIVING_SPEED = 1.07
# urban 35mile/hour = 0.58 mile/min
DRIVING_SPEED = 0.92
# DRIVING_SPEED = 0.6

# # Vehicle driving range in m/fulltank
# VEHICLE_DRIVING_RANGE = 450000
# # 10kw for L2 charger and 64kwh vehicle battery = 6.4 hours
# RECHARGE_TIME = 64/10
# # Driving speed in m/min (65mile per hour)
# DRIVING_SPEED = 1740

FUEL_MULTIPLIER = 0.2

# range of bpr variables
TOPX = 100000000

#400KWH/50KWH(PERCAR)*VEHICLE_DRIVING_RANGE*10
#MOBILE_CHARGER_CAPACITY = 400/50*VEHICLE_DRIVING_RANGE
COUNT_EXISTING_STATIONS = 0

MIP_FOCUS = 3
MIP_GAP = 0.1
CUTS = 0

#########################################################
## Hazard Constants
#########################################################

HAZARD_SPEED = 1
HAZARD_X = 25.272241007699417
HAZARD_Y = -2.3293011832263004
# HAZARD_X = 15.78173465918843
# HAZARD_Y = -2.1863559804326496
HAZARD = [HAZARD_X,HAZARD_Y]
SAFE_BOUNDRY = 5

# HAZARD_X = 5
# HAZARD_Y = 1
# HAZARD = [HAZARD_X,HAZARD_Y]
# SAFE_BOUNDRY = 3

#########################################################
## SUMO_LIB constants
#########################################################
INITIAL_ENERGY = str(64 * FUEL_MULTIPLIER * 1000)  # in Wh

RECHARGE_TIME_MILE_PER_S = 5*38/3600

SIMULATION_FLOW_BEGIN_TIME = "0"

KIA_SOUR_EV_2020_CONSTRAINTS = {
    "minGap": "2.50",
    "maxSpeed": "29.06",
    # "maxSpeed unit is m/s"
    # "maxSpeed": "25",
    "accel": "20.0",
    "decel": "20.0",
    "sigma": "0.0",
    "emissionClass": "Energy/unknown",
    "mass": "1830",
}

KIA_SOUR_EV_2020_PARAMS = {
    "has.battery.device": "true",
    "airDragCoefficient": "0.35",
    "constantPowerIntake": "100",
    "frontSurfaceArea": "2.6",
    "rotatingMass": "40",
    "maximumPower": "150000",
    "propulsionEfficiency": ".98",
    "radialDragCoefficient": "0.1",
    "recuperationEfficiency": ".96",
    "rollDragCoefficient": "0.01",
    "stoppingThreshold": "0.1",
    "device.battery.capacity": "64000",
}
