"""SVG and HTML views of reporting tables."""

from __future__ import annotations

import json
import math
from collections import Counter
from html import escape
from typing import Any, Mapping

from evac.errors import ValidationError
from evac.reporting import ReportColumn, ReportingViews, ReportTable

_COLORS = ("#1261a0", "#b6422c", "#39814a", "#7b4ab1", "#a66b00")


def _require_columns(table: ReportTable, *names: str) -> None:
    if not isinstance(table, ReportTable):
        raise ValidationError("Visualization requires typed ReportTable inputs.")
    available = {column.name for column in table.columns}
    missing = set(names) - available
    if missing:
        raise ValidationError(
            f"Visualization table {table.name!r} is missing columns {sorted(missing)!r}."
        )


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{label} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite.")
    return result


def _document(title: str, body: str, *, height: int = 400) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{height}" '
        f'viewBox="0 0 640 {height}">\n'
        f'<rect width="640" height="{height}" fill="white"/>\n'
        f'<text x="20" y="24" font-family="sans-serif" font-size="14">'
        f"{escape(title)}</text>\n{body}\n</svg>\n"
    )


def line_svg(
    table: ReportTable,
    *,
    x: str,
    y: str,
    group: str | None = None,
    title: str = "EVac view",
) -> str:
    """Render one deterministic SVG line view."""

    _require_columns(table, x, y, *((group,) if group is not None else ()))
    rows: list[tuple[float, float, str]] = []
    for record in table.records():
        x_value = _number(record[x], label=f"{table.name}.{x}")
        y_value = _number(record[y], label=f"{table.name}.{y}")
        rows.append((x_value, y_value, "" if group is None else str(record[group])))
    if not rows:
        raise ValidationError("line_svg cannot render an empty table.")
    min_x, max_x = min(row[0] for row in rows), max(row[0] for row in rows)
    min_y, max_y = min(row[1] for row in rows), max(row[1] for row in rows)
    x_span = max_x - min_x or 1.0
    y_span = max_y - min_y or 1.0
    groups: dict[str, list[tuple[float, float]]] = {}
    for x_value, y_value, group_value in rows:
        groups.setdefault(group_value, []).append((x_value, y_value))
    polylines: list[str] = []
    for index, group_value in enumerate(sorted(groups)):
        points: list[str] = []
        for x_value, y_value in sorted(groups[group_value]):
            pixel_x = 56.0 + 548.0 * (x_value - min_x) / x_span
            pixel_y = 340.0 - 292.0 * (y_value - min_y) / y_span
            points.append(f"{pixel_x:.3f},{pixel_y:.3f}")
        polylines.append(
            f'<polyline fill="none" stroke="{_COLORS[index % len(_COLORS)]}" '
            f'stroke-width="2" points="{" ".join(points)}"/>'
        )
    axes = (
        '<line x1="56" y1="340" x2="604" y2="340" stroke="#222"/>\n'
        '<line x1="56" y1="48" x2="56" y2="340" stroke="#222"/>'
    )
    return _document(title, axes + "\n" + "\n".join(polylines))


def event_metric_svg(
    table: ReportTable,
    *,
    metric: str,
    subject: str,
    title: str | None = None,
) -> str:
    """Plot a numeric value from timeline events."""

    _require_columns(table, "time_seconds", subject, "values")
    rows: list[tuple[Any, ...]] = []
    for record in table.records():
        values = record["values"]
        if not isinstance(values, Mapping):
            raise ValidationError(f"{table.name}.values must be a mapping.")
        if metric in values and values[metric] is not None:
            rows.append((record["time_seconds"], values[metric], record[subject]))
    if not rows:
        raise ValidationError(
            f"Timeline table {table.name!r} has no non-null metric {metric!r}."
        )
    metric_table = ReportTable(
        f"{table.name}.{metric}",
        (
            ReportColumn("time_seconds", "s"),
            ReportColumn(metric),
            ReportColumn(subject),
        ),
        tuple(rows),
    )
    return line_svg(
        metric_table,
        x="time_seconds",
        y=metric,
        group=subject,
        title=title or f"{table.name}: {metric}",
    )


def event_timeline_svg(
    table: ReportTable,
    *,
    subject: str,
    title: str = "Event timeline",
) -> str:
    """Render point events on one row per stable Domain subject."""

    _require_columns(table, "time_seconds", "event_kind", subject)
    records = list(table.records())
    if not records:
        raise ValidationError("event_timeline_svg cannot render an empty table.")
    subjects = sorted({str(record[subject]) for record in records})
    subject_index = {value: index for index, value in enumerate(subjects)}
    times = [
        _number(record["time_seconds"], label="timeline time_seconds")
        for record in records
    ]
    minimum, maximum = min(times), max(times)
    span = maximum - minimum or 1.0
    height = max(240, 92 + 32 * len(subjects))
    bottom = height - 42
    body = [f'<line x1="120" y1="{bottom}" x2="610" y2="{bottom}" stroke="#222"/>']
    for value, index in subject_index.items():
        y = 58 + index * 32
        body.append(
            f'<text x="8" y="{y + 4}" font-family="sans-serif" font-size="11">'
            f"{escape(value)}</text>"
        )
        body.append(f'<line x1="120" y1="{y}" x2="610" y2="{y}" stroke="#ddd"/>')
    for record in sorted(
        records,
        key=lambda item: (
            _number(item["time_seconds"], label="timeline time_seconds"),
            str(item[subject]),
            str(item["event_kind"]),
        ),
    ):
        time_value = _number(record["time_seconds"], label="timeline time_seconds")
        x = 120 + 490 * (time_value - minimum) / span
        y = 58 + subject_index[str(record[subject])] * 32
        body.append(
            f'<circle cx="{x:.3f}" cy="{y}" r="5" fill="#1261a0">'
            f"<title>{escape(str(record['event_kind']))} at {time_value:g} s</title></circle>"
        )
    return _document(title, "\n".join(body), height=height)


def interval_timeline_svg(
    table: ReportTable,
    *,
    subject: str,
    start: str = "start_seconds",
    end: str = "end_seconds",
    label: str | None = None,
    title: str = "Deployment timeline",
) -> str:
    """Render deterministic Gantt-style intervals."""

    _require_columns(
        table, subject, start, end, *((label,) if label is not None else ())
    )
    records = list(table.records())
    if not records:
        raise ValidationError("interval_timeline_svg cannot render an empty table.")
    subjects = sorted({str(record[subject]) for record in records})
    starts = [
        _number(record[start], label=f"{table.name}.{start}") for record in records
    ]
    # An open interval (end is None) has no declared end; it is rendered as
    # reaching the rightmost point known anywhere on this chart.
    ends = [
        None
        if record[end] is None
        else _number(record[end], label=f"{table.name}.{end}")
        for record in records
    ]
    if any(
        right is not None and right <= left
        for left, right in zip(starts, ends, strict=True)
    ):
        raise ValidationError("Timeline intervals must have positive duration.")
    minimum = min(starts)
    maximum = max((*starts, *(value for value in ends if value is not None)))
    span = maximum - minimum or 1.0
    height = max(240, 92 + 34 * len(subjects))
    body: list[str] = []
    for index, value in enumerate(subjects):
        y = 50 + index * 34
        body.append(
            f'<text x="8" y="{y + 15}" font-family="sans-serif" font-size="11">'
            f"{escape(value)}</text>"
        )
    for record in sorted(
        records,
        key=lambda item: (
            str(item[subject]),
            item[start],
            (item[end] is None, item[end] or 0.0),
        ),
    ):
        left = _number(record[start], label=f"{table.name}.{start}")
        open_ended = record[end] is None
        right = (
            maximum if open_ended else _number(record[end], label=f"{table.name}.{end}")
        )
        x = 120 + 490 * (left - minimum) / span
        width = max(1.0, 490 * (right - left) / span)
        y = 50 + subjects.index(str(record[subject])) * 34
        tooltip = str(record[subject]) if label is None else str(record[label])
        right_label = "open" if open_ended else f"{right:g}"
        body.append(
            f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="20" '
            f'fill="#1261a0"><title>{escape(tooltip)}: {left:g}–{right_label} s</title></rect>'
        )
    return _document(title, "\n".join(body), height=height)


def route_usage_svg(
    table: ReportTable,
    *,
    title: str = "Route use",
) -> str:
    """Plot vehicle counts by path identifier."""

    _require_columns(table, "path_id")
    counts = Counter(str(record["path_id"]) for record in table.records())
    if not counts:
        raise ValidationError("route_usage_svg cannot render an empty table.")
    paths = sorted(counts)
    maximum = max(counts.values())
    bar_height = min(24, 280 / len(paths))
    body: list[str] = []
    for index, path_id in enumerate(paths):
        y = 52 + index * (bar_height + 6)
        width = 430 * counts[path_id] / maximum
        body.append(
            f'<text x="8" y="{y + bar_height * 0.75:.3f}" font-family="sans-serif" '
            f'font-size="10">{escape(path_id)}</text>'
        )
        body.append(
            f'<rect x="180" y="{y:.3f}" width="{width:.3f}" height="{bar_height:.3f}" '
            f'fill="#1261a0"><title>{counts[path_id]} vehicles</title></rect>'
        )
    height = max(180, int(80 + len(paths) * (bar_height + 6)))
    return _document(title, "\n".join(body), height=height)


def map_svg(
    nodes: ReportTable,
    edges: ReportTable,
    *,
    path_edges: ReportTable | None = None,
    route_usage: ReportTable | None = None,
    title: str = "Evacuation map",
    origin_label: str = "Evacuation origin",
    destination_label: str = "Destination",
) -> str:
    """Render network geometry, demand roles, chargers, and optional route use.

    Node counts and charging capacity come from ``from_prepared``. Role labels
    describe the caller's scenario; they do not define a hazard model or boundary.
    Tables containing only node identifiers and coordinates remain supported.
    """

    _require_columns(nodes, "node_id", "x", "y")
    _require_columns(edges, "edge_id", "source_node_id", "target_node_id")
    node_records = list(nodes.records())
    if not node_records:
        raise ValidationError("map_svg requires at least one map node.")
    coordinates: dict[str, tuple[float, float]] = {}
    for record in node_records:
        node_id = str(record["node_id"])
        if node_id in coordinates:
            raise ValidationError("map_svg received duplicate node identifiers.")
        coordinates[node_id] = (
            _number(record["x"], label="map node x"),
            _number(record["y"], label="map node y"),
        )
    used_edges: Counter[str] = Counter()
    if (path_edges is None) != (route_usage is None):
        raise ValidationError(
            "map_svg route emphasis requires path_edges and route_usage together."
        )
    if path_edges is not None and route_usage is not None:
        _require_columns(path_edges, "path_id", "edge_id")
        _require_columns(route_usage, "path_id")
        path_counts = Counter(
            str(record["path_id"]) for record in route_usage.records()
        )
        for record in path_edges.records():
            used_edges[str(record["edge_id"])] += path_counts[str(record["path_id"])]
    xs = [item[0] for item in coordinates.values()]
    ys = [item[1] for item in coordinates.values()]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    x_span, y_span = max_x - min_x or 1.0, max_y - min_y or 1.0
    # One scale for both axes preserves the supplied local geometry.
    scale = min(510 / x_span, 490 / y_span)
    left = 48 + (510 - x_span * scale) / 2
    top = 82 + (490 - y_span * scale) / 2

    def point(node_id: str) -> tuple[float, float]:
        if node_id not in coordinates:
            raise ValidationError(f"Map edge references unknown node {node_id!r}.")
        x, y = coordinates[node_id]
        return left + (x - min_x) * scale, top + (max_y - y) * scale

    body: list[str] = []
    maximum_use = max(used_edges.values(), default=1)
    for record in sorted(edges.records(), key=lambda item: str(item["edge_id"])):
        edge_id = str(record["edge_id"])
        x1, y1 = point(str(record["source_node_id"]))
        x2, y2 = point(str(record["target_node_id"]))
        use = used_edges[edge_id]
        color = "#705a9b" if use else "#aeb7bc"
        width = 1.5 + (4.5 * use / maximum_use if use else 0.0)
        body.append(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="{color}" stroke-width="{width:.3f}"><title>{escape(edge_id)}'
            f"{f': {use} route uses' if use else ''}</title></line>"
        )
    has_origins = has_destinations = has_chargers = has_sites = False
    for record in node_records:
        node_id = str(record["node_id"])
        x, y = point(node_id)
        origin_count = record.get("origin_vehicle_count", 0)
        destination_count = record.get("destination_vehicle_count", 0)
        ports = record.get("fixed_charger_ports", 0)
        capacity = record.get("mcs_limit", 0)
        has_origins |= origin_count > 0
        has_destinations |= destination_count > 0
        has_chargers |= ports > 0
        has_sites |= capacity > 0
        body.append(f'<g aria-label="Node {escape(node_id)}">')
        if capacity:
            body.append(
                f'<circle cx="{x:.3f}" cy="{y:.3f}" r="16" fill="none" '
                f'stroke="#ad7a21" stroke-width="1.5" stroke-dasharray="4 3">'
                f'<title>Eligible MCS site: up to {capacity} units</title></circle>'
            )
        if origin_count:
            offset = -7 if destination_count else 0
            center = x + offset
            body.append(
                f'<path d="M {center:.3f},{y - 9:.3f} l 9,17 h -18 Z" '
                f'fill="#b64b45" stroke="white" stroke-width="1.5">'
                f'<title>{escape(origin_label)}: {origin_count} EVs</title></path>'
            )
        if destination_count:
            offset = 7 if origin_count else 0
            body.append(
                f'<rect x="{x + offset - 7:.3f}" y="{y - 7:.3f}" width="14" height="14" '
                f'fill="#2e7964" stroke="white" stroke-width="1.5">'
                f'<title>{escape(destination_label)}: {destination_count} EVs</title></rect>'
            )
        if not origin_count and not destination_count:
            body.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="#465663"/>')
        body.append(
            f'<text x="{x + 19:.3f}" y="{y + 4:.3f}" font-size="14" '
            f'font-weight="600">{escape(node_id)}</text>'
        )
        if ports:
            # Keep origin-side infrastructure clear of nearby departure roads.
            badge_x, badge_y = (x + 45, y - 6) if origin_count else (x - 36, y + 22)
            body.append(
                f'<rect x="{badge_x:.3f}" y="{badge_y:.3f}" width="12" height="12" '
                f'rx="2" fill="#256a98"/>'
                f'<text x="{badge_x + 6:.3f}" y="{badge_y + 9:.3f}" fill="white" '
                f'font-size="9" text-anchor="middle">C</text>'
                f'<text x="{badge_x + 17:.3f}" y="{badge_y + 10:.3f}" fill="#256a98" '
                f'font-size="13" paint-order="stroke" stroke="white" stroke-width="3">{ports} ports</text>'
            )
        if origin_count or destination_count:
            count_label = (
                f"{origin_count} leaving / {destination_count} arriving"
                if origin_count and destination_count else
                f"{origin_count or destination_count} EVs"
            )
            body.append(
                f'<text x="{x:.3f}" y="{y - 25:.3f}" font-size="13" '
                f'text-anchor="middle" paint-order="stroke" stroke="white" '
                f'stroke-width="4">{count_label}</text>'
            )
        body.append('</g>')
    body.append('<text x="610" y="95" font-size="17" font-weight="600">Reading the network</text>')
    legend_y = 136
    entries = []
    if has_origins:
        entries.append(("origin", origin_label, "Vehicles leave from here."))
    if has_destinations:
        entries.append(("destination", destination_label, "Vehicles finish their journeys here."))
    if has_chargers:
        entries.append(("charger", "Fixed charging station", "The number gives charging ports."))
    if has_sites:
        entries.append(("mcs", "Eligible MCS site", "A possible location, not a deployment."))
    entries.append(("road", "Road connection", "Directions and distances are in the inputs."))
    if route_usage is not None:
        entries.append(("used", "Road used by the plan", "Thicker lines carry more planned vehicles."))
    for kind, label, detail in entries:
        if kind == "origin":
            symbol = f'<path d="M 620,{legend_y - 10} l 8,16 h -16 Z" fill="#b64b45"/>'
        elif kind == "destination":
            symbol = f'<rect x="613" y="{legend_y - 7}" width="14" height="14" fill="#2e7964"/>'
        elif kind == "charger":
            symbol = f'<rect x="613" y="{legend_y - 7}" width="14" height="14" rx="2" fill="#256a98"/><text x="620" y="{legend_y + 4}" text-anchor="middle" fill="white" font-size="11">C</text>'
        elif kind == "mcs":
            symbol = f'<circle cx="620" cy="{legend_y}" r="11" fill="none" stroke="#ad7a21" stroke-width="1.5" stroke-dasharray="4 3"/>'
        else:
            color = "#705a9b" if kind == "used" else "#aeb7bc"
            symbol = f'<line x1="610" y1="{legend_y}" x2="630" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        body.append(symbol)
        body.append(f'<text x="644" y="{legend_y + 4}" font-size="14" font-weight="600">{escape(label)}</text>')
        body.append(f'<text x="644" y="{legend_y + 25}" font-size="11" fill="#5b6770">{escape(detail)}</text>')
        legend_y += 66
    body.append('<text x="48" y="625" font-size="12" fill="#5b6770">Supplied node geometry; lines connect network nodes and do not trace road alignments.</text>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="650" viewBox="0 0 960 650" '
        'role="img" aria-labelledby="map-title" style="max-width:100%;height:auto" '
        'font-family="Arial, sans-serif" fill="#263745">'
        f'<title id="map-title">{escape(title)}</title>'
        '<rect width="960" height="650" fill="white"/>'
        f'<text x="36" y="34" font-size="22" font-weight="600">{escape(title)}</text>'
        + "\n".join(body) + '</svg>\n'
    )


def animation_html(
    views: ReportingViews,
    *,
    title: str = "EVac event animation",
) -> str:
    """Build a reusable event-step animation from authoritative Reporting views."""

    if not isinstance(views, ReportingViews):
        raise ValidationError("animation_html requires typed ReportingViews.")
    sources = (
        ("vehicle_timeline", "vehicle_id"),
        ("charging_site_timeline", "charging_site_id"),
        ("edge_usage", "edge_id"),
        ("mcs_timeline", "mcs_unit_id"),
    )
    events: list[dict[str, Any]] = []
    for table_name, subject_column in sources:
        if table_name not in views.tables:
            continue
        table = views.table(table_name)
        _require_columns(table, "time_seconds", "event_kind", subject_column, "values")
        for record in table.records():
            events.append(
                {
                    "time_seconds": _number(
                        record["time_seconds"], label=f"{table_name}.time_seconds"
                    ),
                    "kind": str(record["event_kind"]),
                    "subject_namespace": table_name,
                    "subject": str(record[subject_column]),
                    "values": dict(record["values"]),
                }
            )
    if not events:
        raise ValidationError("animation_html requires at least one reporting event.")
    events.sort(
        key=lambda item: (
            item["time_seconds"],
            item["subject_namespace"],
            item["subject"],
            item["kind"],
        )
    )
    try:
        payload = json.dumps(events, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "Reporting event values must be JSON-compatible for animation."
        ) from exc
    payload = payload.replace("<", "\\u003c")
    heading = escape(title)
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>{heading}</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 24px; color: #202124; }}
#event {{ padding: 16px; border: 1px solid #dadce0; border-radius: 8px; white-space: pre-wrap; }}
input {{ width: min(720px, 100%); }}
</style>
<h1>{heading}</h1>
<input id="step" type="range" min="0" max="{len(events) - 1}" value="0">
<p id="clock"></p><div id="event"></div>
<script>
const events = {payload};
const step = document.getElementById("step");
const clock = document.getElementById("clock");
const output = document.getElementById("event");
function render() {{
  const event = events[Number(step.value)];
  clock.textContent = `Event ${{Number(step.value) + 1}} / ${{events.length}} — ${{event.time_seconds}} s`;
  output.textContent = `${{event.subject_namespace}} / ${{event.subject}} / ${{event.kind}}\n` +
    JSON.stringify(event.values, null, 2);
}}
step.addEventListener("input", render); render();
</script>
</html>
"""


__all__ = [
    "animation_html",
    "event_metric_svg",
    "event_timeline_svg",
    "interval_timeline_svg",
    "line_svg",
    "map_svg",
    "route_usage_svg",
]
