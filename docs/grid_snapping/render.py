"""Render measured executable/candidate outputs without rerunning either algorithm."""

import html
import json


def edge_set(edges):
    return {tuple(map(tuple, edge)) for edge in edges}


def cell_centers(record):
    if not record["xnodes"] or not record["ynodes"]:
        return []
    x_centers = [0.5 * (a + b) for a, b in zip(record["xnodes"], record["xnodes"][1:])]
    y_centers = [0.5 * (a + b) for a, b in zip(record["ynodes"], record["ynodes"][1:])]
    return [(x, y) for y in y_centers for x in x_centers]


def has_figure(record, records):
    """Every case that differs from the executable is drawn, or the only case."""
    if record["error"] is not None:
        return False
    if len(records) == 1:
        return True
    return edge_set(record["reference"]) != edge_set(record["candidate"])


def render(record, candidate_label):
    """Draw the actual measured edges; no historical counts or path assumptions."""
    reference, candidate = edge_set(record["reference"]), edge_set(record["candidate"])
    missing, added = reference - candidate, candidate - reference
    xs = record["xnodes"] + [p[0] for line in record["lines"] for p in line]
    ys = record["ynodes"] + [p[1] for line in record["lines"] for p in line]
    xmin, xmax, ymin, ymax = min(xs) - .3, max(xs) + .3, min(ys) - .3, max(ys) + .3
    scale = min(300 / (xmax - xmin), 310 / (ymax - ymin))
    cell_points = cell_centers(record)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500" viewBox="0 0 1200 500" role="img">',
        f'<title>{html.escape(record["name"])}</title>',
        f'<desc>Actual iMOD5 executable versus {html.escape(candidate_label)}.</desc>',
        '<rect width="1200" height="500" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172b4d;font-size:12px}'
        '.input{stroke:#353c47;stroke-width:2;stroke-dasharray:6 4;fill:none}</style>',
        f'<text x="24" y="30" style="font-size:21px">{html.escape(record["name"])}</text>',
        f'<text x="24" y="55">Missing {len(missing)}, added {len(added)} (unique geometric walls)</text>',
    ]
    panels = [
        ("iMOD5 executable", [(reference, "#1264a3")], len(reference)),
        (candidate_label, [(candidate, "#c26b00")], len(candidate)),
        ("Difference", [(reference & candidate, "#b8c3ce"), (missing, "#c21d42"),
                        (added, "#7c3aed")], None),
    ]
    for panel, (label, groups, count) in enumerate(panels):
        offset = 32 + panel * 400
        left = offset + (300 - scale * (xmax - xmin)) / 2
        top = 116 + (310 - scale * (ymax - ymin)) / 2

        def xy(point):
            return left + (point[0] - xmin) * scale, top + (ymax - point[1]) * scale

        suffix = "" if count is None else f": {count} edges"
        parts.append(
            f'<text x="{offset}" y="90" style="font-size:16px;font-weight:bold">'
            f'{html.escape(label + suffix)}</text>'
        )
        for x in record["xnodes"]:
            x0, y0 = xy((x, min(record["ynodes"])))
            x1, y1 = xy((x, max(record["ynodes"])))
            parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#dde3eb"/>')
            parts.append(f'<text x="{x0}" y="{y0+18}" text-anchor="middle">{x:g}</text>')
        for y in record["ynodes"]:
            x0, y0 = xy((min(record["xnodes"]), y))
            x1, y1 = xy((max(record["xnodes"]), y))
            parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#dde3eb"/>')
            parts.append(f'<text x="{x0-8}" y="{y0+4}" text-anchor="end">{y:g}</text>')
        for cx, cy in cell_points:
            px, py = xy((cx, cy))
            parts.append(f'<circle cx="{px}" cy="{py}" r="2.5" fill="#5f6f83" stroke="#5f6f83" opacity="0.9"/>')
        for edges, colour in groups:
            for first, second in sorted(edges):
                x0, y0 = xy(first)
                x1, y1 = xy(second)
                parts.append(
                    f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
                    f'stroke="{colour}" stroke-width="5" stroke-linecap="round"/>'
                )
        for line in record["lines"]:
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(xy, line))
            parts.append(f'<polyline points="{coordinates}" class="input"/>')
            for point, fill in ((line[0], "white"), (line[-1], "#353c47")):
                x, y = xy(point)
                parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{fill}" stroke="#353c47"/>')

    parts.extend([
        '<g font-size="12">',
        '<rect x="28" y="430" width="520" height="56" fill="#f8fafc" stroke="#dfe5ee" rx="8"/>',
        '<line x1="42" y1="445" x2="74" y2="445" stroke="#1264a3" stroke-width="4"/>',
        '<text x="82" y="449">iMOD5 executable</text>',
        '<line x1="210" y1="445" x2="242" y2="445" stroke="#c26b00" stroke-width="4"/>',
        '<text x="250" y="449">' + html.escape(candidate_label) + '</text>',
        '<line x1="42" y1="462" x2="74" y2="462" stroke="#c21d42" stroke-width="4"/>',
        '<text x="82" y="466">Missing</text>',
        '<line x1="210" y1="462" x2="242" y2="462" stroke="#7c3aed" stroke-width="4"/>',
        '<text x="250" y="466">Added</text>',
        '<circle cx="365" cy="462" r="3.5" fill="#5f6f83" stroke="#5f6f83"/>',
        '<text x="375" y="466">Cell centers</text>',
        '</g>',
        '<text x="32" y="465">Dashed: input; open=start, filled=end</text>',
        '</svg>',
    ])
    return "\n".join(parts)


def write_report(records, destination, candidate_label, metadata):
    """Build Markdown, self-contained HTML, SVG figures and raw measured results."""
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for stale in figures.glob("*.svg"):
        stale.unlink()
    differing = sum(has_figure(record, records) for record in records)
    title = f"iMOD5 executable versus {candidate_label}"
    method = (
        "Reference edges come from actual iMOD5 executable GENSNAPTOGRID output, "
        "not a Python port. Candidate edges come from "
        + ("this repository's public snap_gen_to_grid implementation."
           if candidate_label == "Repository implementation" else
           "unadjusted xugrid.create_snap_to_grid_dataframe, max_snap_distance=0.0, "
           "default tolerance=1e-12; no outside guard.")
        + " Both use the same serialized GEN input (six decimal places) and grid. "
        "Results compare combined unique geometric walls, NOT per-feature assignments. "
        "Direction, attributes, row order and 3D output are not assessed. "
        "Dense random samples can hide per-feature differences. These are synthetic "
        "examples, not hydrological validation. Random perturbations use seed 43; "
        "tie-heavy inputs use seed 42. Random feature 97 is run in isolation, so "
        "iMOD5's final-feature rules apply to it."
    )
    markdown = [
        f"# {title}", "", method, "",
        "[Self-contained HTML gallery](known_differences.html) | [Measured data](results.json)",
        "", "Missing = executable-only; added = candidate-only. "
        f"{differing} of {len(records)} cases differ from the executable, and each of "
        "them is drawn under Figures. A library error is "
        "reported explicitly, never counted as a successful empty result.", "",
        "Blue: iMOD5 executable. Orange: " + candidate_label
        + ". Difference panel: grey = shared, red = missing (iMOD5 executable-only), "
        + "purple = added (candidate-only), small grey dots = cell centers.",
    ]
    markdown_rows, rows, sections = [], [], []
    equal_cases = different_cases = equal_edges = different_edges = 0
    for record in records:
        reference = edge_set(record["reference"])
        if record["error"] is not None:
            values = [record["name"], str(len(reference)), "ERROR: " + record["error"], "-", "-"]
        else:
            candidate = edge_set(record["candidate"])
            missing, added = reference - candidate, candidate - reference
            equal_cases += reference == candidate
            different_cases += reference != candidate
            equal_edges += len(reference & candidate)
            different_edges += len(missing) + len(added)
            values = [record["name"], str(len(reference)), str(len(candidate)),
                      str(len(missing)), str(len(added))]
        markdown_rows.append("| " + " | ".join(values) + " |")
        rows.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
        if has_figure(record, records):
            svg = render(record, candidate_label)
            (figures / f'{record["name"]}.svg').write_text(svg, encoding="utf-8")
            sections.append(
                f'<section id="{record["name"]}"><h2>{html.escape(record["name"])}</h2>'
                + svg + "</section>"
            )
    summary = (
        f"Cases equal: {equal_cases}; cases different: {different_cases}; "
        f"snapped edges equal: {equal_edges}; snapped edges different: {different_edges}."
    )
    summary_method = (
        "Totals exclude cases with errors. Equal edges are shared walls; different "
        "edges are missing plus added walls, summed across cases."
    )
    markdown.extend([
        "", summary, "", summary_method, "",
        "| Case | iMOD5 executable walls | " + candidate_label + " walls | Missing | Added |",
        "|---|---:|---:|---:|---:|",
        *markdown_rows,
    ])
    markdown.extend(["", "## Figures", ""])
    for record in records:
        if has_figure(record, records):
            markdown.extend([
                f'### {record["name"]}', "",
                f'![{title}: {record["name"]}](figures/{record["name"]}.svg)', "",
            ])
    markdown.extend([
        "## Versions and provenance", "", "```json",
        json.dumps(metadata, indent=2), "```", "",
        "The main repository comparison is rebuilt with `pixi run compare-grid-snapping`.",
        "The separate `default_xugrid` report is a one-off snapshot and is not updated "
        "by that task.", "",
    ])
    headers = ("Case", "iMOD5 executable walls", candidate_label + " walls", "Missing", "Added")
    page = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        "<style>body{font:16px/1.5 system-ui;max-width:1250px;margin:auto;padding:24px;"
        "color:#172b4d;background:#f3f6fb}section{background:white;padding:16px;margin:24px 0}"
        "svg{width:100%;height:auto}table{border-collapse:collapse}"
        "td,th{padding:6px;border-bottom:1px solid #ccc;text-align:left}</style></head><body>",
        f"<h1>{html.escape(title)}</h1><p>{html.escape(method)}</p>",
        '<p><a href="known_differences.md">Markdown report</a> | '
        '<a href="results.json">Measured data</a></p>',
        "<p>Blue: iMOD5 executable. Orange: " + html.escape(candidate_label)
        + ". Red: missing (iMOD5 executable-only). Purple: added (candidate-only). "
        + "Grey: shared. Small grey dots: cell centers.</p>",
        f"<p>{html.escape(summary)}</p><p>{html.escape(summary_method)}</p>",
        "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>",
        *sections,
        "<h2>Versions and provenance</h2><pre>" + html.escape(json.dumps(metadata, indent=2))
        + "</pre></body></html>",
    ]
    (destination / "known_differences.md").write_text("\n".join(markdown), encoding="utf-8")
    (destination / "known_differences.html").write_text("\n".join(page), encoding="utf-8")
    measurements = []
    for record in records:
        if has_figure(record, records):
            measurements.append(record)
            continue
        reference = edge_set(record["reference"])
        candidate = None if record["candidate"] is None else edge_set(record["candidate"])
        measurements.append({
            "name": record["name"],
            "reference_walls": len(reference),
            "candidate_walls": None if candidate is None else len(candidate),
            "missing": None if candidate is None else sorted(reference - candidate),
            "added": None if candidate is None else sorted(candidate - reference),
            "error": record["error"],
        })
    (destination / "results.json").write_text(
        json.dumps({"candidate": candidate_label, "metadata": metadata, "cases": measurements}, indent=2)
        + "\n", encoding="utf-8",
    )
