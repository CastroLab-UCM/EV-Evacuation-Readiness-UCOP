from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from evac.domain import PlannerTermination, PreparedScenario, RunResult, TimelineEvent
from evac.errors import ValidationError


@dataclass(frozen=True, slots=True)
class ReportColumn:
    name: str
    unit: str | None = None
    nullable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("ReportColumn.name must be non-empty.")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
            raise ValidationError(
                "ReportColumn.unit must be a non-empty string when set."
            )
        if not isinstance(self.nullable, bool):
            raise ValidationError("ReportColumn.nullable must be boolean.")


@dataclass(frozen=True, slots=True)
class ReportTable:
    name: str
    columns: tuple[ReportColumn, ...]
    rows: tuple[tuple[Any, ...], ...]
    primary_key: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("ReportTable.name must be non-empty.")
        names = tuple((column.name for column in self.columns))
        if not names or len(set(names)) != len(names):
            raise ValidationError("ReportTable columns must be non-empty and unique.")
        if any((key not in names for key in self.primary_key)):
            raise ValidationError(
                "ReportTable primary key references an unknown column."
            )
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValidationError(
                    f"ReportTable {self.name!r} row has invalid cardinality."
                )
            for column, value in zip(self.columns, row, strict=True):
                if value is None and (not column.nullable):
                    raise ValidationError(
                        f"ReportTable {self.name!r} column {column.name!r} is not nullable."
                    )

    def records(self) -> tuple[Mapping[str, Any], ...]:
        names = tuple((column.name for column in self.columns))
        return tuple(
            (MappingProxyType(dict(zip(names, row, strict=True))) for row in self.rows)
        )

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ValidationError(
                "ReportTable.to_dataframe requires the optional pandas dependency."
            ) from exc
        return pd.DataFrame(
            self.records(), columns=[column.name for column in self.columns]
        )


@dataclass(frozen=True, slots=True)
class ReportingViews:
    tables: Mapping[str, ReportTable]

    def __post_init__(self) -> None:
        if any((name != table.name for name, table in self.tables.items())):
            raise ValidationError("ReportingViews keys must equal their table names.")
        object.__setattr__(self, "tables", MappingProxyType(dict(self.tables)))

    def table(self, name: str) -> ReportTable:
        try:
            return self.tables[name]
        except KeyError as exc:
            raise ValidationError(f"Unknown reporting table {name!r}.") from exc

    def save_csv(self, directory):
        """Save each public table as a CSV, retaining explicit unit-bearing column names."""
        from pathlib import Path

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in self.tables.items():
            table.to_dataframe().to_csv(destination / f"{name}.csv", index=False)


def _timeline_table(
    name: str, subject_column: str, events: Iterable[TimelineEvent]
) -> ReportTable:
    rows = tuple(
        (
            (
                event.time_seconds,
                event.kind,
                event.subject_namespace,
                event.subject_value,
                dict(event.values),
            )
            for event in sorted(
                events,
                key=lambda event: (
                    event.time_seconds,
                    event.subject_namespace,
                    str(event.subject_value),
                    event.kind,
                ),
            )
        )
    )
    return ReportTable(
        name,
        (
            ReportColumn("time_seconds", "s"),
            ReportColumn("event_kind"),
            ReportColumn("subject_namespace"),
            ReportColumn(subject_column),
            ReportColumn("values"),
        ),
        rows,
    )


def from_result(result: RunResult) -> ReportingViews:
    """Expose solver status, scientific metrics, and the returned plan."""
    planner, evaluation, simulation, plan = (
        result.planner,
        result.evaluation,
        result.simulation,
        result.plan,
    )
    summary_values = {
        "solver_status": planner.termination.value,
        "has_plan": plan is not None,
        "proven_optimal": planner.termination is PlannerTermination.OPTIMAL,
        "simulation_feasible": None if evaluation is None else evaluation.feasible,
        "objective": None if evaluation is None else evaluation.objective_name,
        "formulation_objective_seconds": planner.formulation_objective,
        "formulation_bound_seconds": planner.objective_bound,
        "relative_gap": planner.objective_gap,
        "simulated_objective_seconds": None
        if evaluation is None
        else evaluation.scientific_objective,
        "terminal_cause": planner.terminal_cause,
        **({} if evaluation is None else dict(evaluation.metrics)),
        **dict(result.timings),
    }
    summary = ReportTable(
        "summary",
        (ReportColumn("metric"), ReportColumn("value", nullable=True)),
        tuple(summary_values.items()),
        ("metric",),
    )
    vehicle_events = () if simulation is None else simulation.vehicle_events
    edge_events = () if simulation is None else simulation.edge_events
    charging_events = () if simulation is None else simulation.charging_site_events
    mcs_events = () if simulation is None else simulation.mcs_events
    arrivals = sorted(
        (event for event in vehicle_events if event.kind == "vehicle_arrived"),
        key=lambda event: (event.time_seconds, str(event.subject_value)),
    )
    completion = ReportTable(
        "evacuation_completion_series",
        (
            ReportColumn("time_seconds", "s"),
            ReportColumn("arrived_vehicle_count", "vehicles"),
        ),
        tuple(
            (
                (event.time_seconds, index)
                for index, event in enumerate(arrivals, start=1)
            )
        ),
    )
    route_usage = ReportTable(
        "route_usage",
        (
            ReportColumn("vehicle_id"),
            ReportColumn("path_id"),
            ReportColumn("departure_seconds", "s"),
        ),
        ()
        if plan is None
        else tuple(
            (
                (item.vehicle_id.value, item.path_id.value, item.departure_seconds)
                for item in plan.vehicle_plans
            )
        ),
        ("vehicle_id",),
    )
    deployments = ReportTable(
        "mcs_deployments",
        (
            ReportColumn("deployment_id"),
            ReportColumn("mcs_unit_id"),
            ReportColumn("site_id"),
        ),
        ()
        if plan is None
        else tuple(
            (
                (
                    item.id.value,
                    item.mcs_unit_id.value,
                    item.site_id.value,
                )
                for item in plan.mcs_deployments
            )
        ),
        ("deployment_id",),
    )
    {
        event.subject_value: event.time_seconds
        for event in vehicle_events
        if event.kind == "vehicle_departed"
    }
    schedule = ReportTable(
        "schedule",
        (
            ReportColumn("vehicle_id"),
            ReportColumn("departure_seconds", "s"),
            ReportColumn("completion_seconds", "s", nullable=True),
            ReportColumn("evacuation_duration_seconds", "s", nullable=True),
        ),
        ()
        if plan is None
        else tuple(
            (
                (
                    item.vehicle_id.value,
                    item.departure_seconds,
                    next(
                        (
                            event.time_seconds
                            for event in arrivals
                            if event.subject_value == item.vehicle_id.value
                        ),
                        None,
                    ),
                    next(
                        (
                            event.time_seconds - item.departure_seconds
                            for event in arrivals
                            if event.subject_value == item.vehicle_id.value
                        ),
                        None,
                    ),
                )
                for item in plan.vehicle_plans
            )
        ),
        ("vehicle_id",),
    )
    tables = (
        summary,
        schedule,
        completion,
        deployments,
        route_usage,
        _timeline_table("vehicle_timeline", "vehicle_id", vehicle_events),
        _timeline_table("charging_site_timeline", "charging_site_id", charging_events),
        _timeline_table("edge_usage", "edge_id", edge_events),
        _timeline_table("mcs_timeline", "mcs_unit_id", mcs_events),
    )
    return ReportingViews({table.name: table for table in tables})


def from_prepared(prepared_scenario: PreparedScenario) -> ReportingViews:
    if not isinstance(prepared_scenario, PreparedScenario):
        raise ValidationError(
            "reporting.from_prepared requires a typed PreparedScenario."
        )
    network = prepared_scenario.network
    origins = Counter(vehicle.origin for vehicle in prepared_scenario.vehicles)
    destinations = Counter(vehicle.destination for vehicle in prepared_scenario.vehicles)
    nodes = ReportTable(
        "map_nodes",
        (
            ReportColumn("node_id"),
            ReportColumn("x", network.coordinate_units),
            ReportColumn("y", network.coordinate_units),
            ReportColumn("charging_provider_count", "providers"),
            ReportColumn("mcs_limit", "MCS units"),
            ReportColumn("fixed_charger_ports", "ports"),
            ReportColumn("origin_vehicle_count", "vehicles"),
            ReportColumn("destination_vehicle_count", "vehicles"),
        ),
        tuple(
            (
                (
                    node.id.value,
                    node.coordinate[0],
                    node.coordinate[1],
                    len(node.charging_providers),
                    node.mcs_limit,
                    sum(provider.port_count for provider in node.charging_providers
                        if provider.kind == "fcs"),
                    origins[node.id],
                    destinations[node.id],
                )
                for node in sorted(network.nodes, key=lambda item: str(item.id.value))
            )
        ),
        ("node_id",),
    )
    edges = ReportTable(
        "map_edges",
        (
            ReportColumn("edge_id"),
            ReportColumn("source_node_id"),
            ReportColumn("target_node_id"),
            ReportColumn("distance_m", "m"),
            ReportColumn("free_flow_duration_seconds", "s"),
            ReportColumn("speed_limit_m_per_second", "m/s", nullable=True),
        ),
        tuple(
            (
                (
                    edge.id.value,
                    edge.source.value,
                    edge.target.value,
                    edge.distance_m,
                    edge.free_flow_duration_seconds,
                    edge.speed_limit_m_per_second,
                )
                for edge in sorted(network.edges, key=lambda item: str(item.id.value))
            )
        ),
        ("edge_id",),
    )
    path_edges = ReportTable(
        "candidate_path_edges",
        (
            ReportColumn("path_id"),
            ReportColumn("edge_sequence"),
            ReportColumn("edge_id"),
        ),
        tuple(
            (
                (path.id.value, sequence, edge_id.value)
                for path in sorted(
                    prepared_scenario.candidate_paths,
                    key=lambda item: str(item.id.value),
                )
                for sequence, edge_id in enumerate(path.edge_ids)
            )
        ),
        ("path_id", "edge_sequence"),
    )
    tables = (nodes, edges, path_edges)
    return ReportingViews({table.name: table for table in tables})


__all__ = [
    "ReportColumn",
    "ReportTable",
    "ReportingViews",
    "from_prepared",
    "from_result",
]
