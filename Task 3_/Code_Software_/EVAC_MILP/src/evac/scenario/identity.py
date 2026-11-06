from __future__ import annotations
import json
from typing import Any
from evac.domain import SemanticIdentity
from evac.scenario.authored import AuthoredScenarioResources
from evac.scenario.codecs import load_scenario_resources
from evac.scenario.models import Scenario
from evac.scenario.policy import normalized_semantic_overlay


def _identifier_key(value: object) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _semantic_resources(value: AuthoredScenarioResources) -> dict[str, Any]:
    authored_map = value.map
    traffic = value.traffic
    traffic_data: dict[str, Any] = {
        "exogenous_profile": traffic.exogenous_profile.to_data()
    }
    return {
        "map": {
            "id": authored_map.id.to_data(),
            "coordinate_system": authored_map.coordinate_system,
            "coordinate_units": authored_map.coordinate_units,
            "directionality": authored_map.directionality.value,
            "nodes": [
                {
                    "id": item.id.to_data(),
                    "coordinate": list(item.coordinate),
                    "fixed_charger_ports": item.fixed_charger_ports,
                    "fixed_charger_power_kw": item.fixed_charger_power_kw,
                }
                for item in sorted(
                    authored_map.nodes, key=lambda item: _identifier_key(item.id.value)
                )
            ],
            "roads": [
                {
                    "id": item.id.to_data(),
                    "source": item.source.to_data(),
                    "target": item.target.to_data(),
                    "distance_m": item.distance_m,
                    "free_flow_duration_seconds": item.free_flow_duration_seconds,
                    "source_speed_m_per_second": item.source_speed_m_per_second,
                }
                for item in sorted(
                    authored_map.roads, key=lambda item: _identifier_key(item.id.value)
                )
            ],
        },
        "vehicles": [
            {
                "id": item.id.to_data(),
                "battery_kwh": item.battery_kwh,
                "efficiency_m_per_kwh": item.efficiency_m_per_kwh,
            }
            for item in sorted(
                value.vehicles.types, key=lambda item: _identifier_key(item.id.value)
            )
        ],
        "mobile_chargers": [
            {
                "id": item.id.to_data(),
                "battery_kwh": item.battery_kwh,
                "power_kw": item.power_kw,
                "port_count": item.port_count,
            }
            for item in sorted(
                value.mobile_chargers.types,
                key=lambda item: _identifier_key(item.id.value),
            )
        ],
        "demand": [
            {
                "id": item.id.to_data(),
                "vehicle_type_id": item.vehicle_type_id.to_data(),
                "speed_m_per_second": item.speed_m_per_second,
                "initial_soc": item.initial_soc,
                "origin": item.origin.to_data(),
                "destination": item.destination.to_data(),
                "size": item.size,
            }
            for item in sorted(
                value.demand, key=lambda item: _identifier_key(item.id.value)
            )
        ],
        "supply": [
            {
                "id": item.id.to_data(),
                "mcs_type_id": item.mcs_type_id.to_data(),
                "initial_soc": item.initial_soc,
            }
            for item in sorted(
                value.supply, key=lambda item: _identifier_key(item.id.value)
            )
        ],
        "traffic": traffic_data,
        "evaluation": {
            "objectives": [
                {
                    "name": item.name,
                    "aggregation": item.aggregation,
                    "norm": item.norm,
                    "unit": item.unit,
                    "weight": item.weight,
                    "priority": item.priority,
                }
                for item in value.evaluation.objectives
            ]
        },
        "overlay": normalized_semantic_overlay(value),
    }


def scenario_identity_from_resources(
    scenario: Scenario, resources: AuthoredScenarioResources
) -> SemanticIdentity:
    value = {
        "id": scenario.id.to_data(),
        "references": [
            {"role": reference.role, "key": reference.key}
            for reference in scenario.resource_references()
        ],
        "resources": _semantic_resources(resources),
    }
    return SemanticIdentity(
        "scenario",
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    )


def scenario_identity(scenario: Scenario) -> SemanticIdentity:
    return scenario_identity_from_resources(scenario, load_scenario_resources(scenario))


__all__ = ["scenario_identity", "scenario_identity_from_resources"]
