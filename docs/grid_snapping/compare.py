"""Rebuild executable-backed snapping comparisons and their reference documents."""

import argparse
import importlib.metadata
import platform
import tempfile
from pathlib import Path

import imod
import numpy as np
import xugrid as xu

from docs.grid_snapping.render import write_report
from src.grid_helpers import (
    create_target_grid,
    grid_spec,
    open_idf_grid,
    x_nodes,
    y_nodes,
)
from src.snapping import snap_gen_to_grid
from tests import test_snapping_imod5 as imod5_tests

HERE = Path(__file__).resolve().parent


def edge_key(endpoints):
    return tuple(sorted(tuple(float(v) for v in point) for point in endpoints))


def output_edges(path):
    """Read actual GEN output, including the executable's explicit empty file."""
    return sorted(
        edge_key(edge)
        for edge in imod5_tests.wall_set(imod5_tests.read_gen_or_empty(path))
    )


def measure(name, lines, arguments, directory, default_xugrid=False):
    """Run the actual executable and candidate on the same serialized input/grid."""
    print(f"Comparing {name}", flush=True)
    directory.mkdir()
    source, reference = [directory / name for name in (
        "input.gen", "imod5.gen",
    )]
    imod5_tests.write_ascii_gen(source, lines)
    process = imod5_tests.run_imod5(directory, source, reference, **arguments)
    if process.returncode != 0 or not reference.exists():
        raise RuntimeError(
            f"iMOD5 failed for {name} (exit {process.returncode}).\n"
            f"{process.stdout}\n{process.stderr}"
        )
    like = open_idf_grid(arguments["idffile"]) if "idffile" in arguments else None
    cellsize = arguments.get("cellsize") if like is None else None
    target = create_target_grid(like, arguments.get("window"), cellsize)
    grid = grid_spec(target)
    geometry = imod.gen.read(source)
    error = None
    if default_xugrid:
        # Use the same GEN precision as the executable, without adapter corrections.
        topology = xu.Ugrid2d.from_structured_intervals1d(x_nodes(grid), y_nodes(grid))
        try:
            frame = xu.create_snap_to_grid_dataframe(
                geometry, topology, max_snap_distance=0.0
            )
        except ValueError as exception:
            # Library errors are comparison outcomes, never successful empty output.
            error = f"{type(exception).__name__}: {exception}"
            edges = None
        else:
            endpoints = topology.node_coordinates[
                topology.edge_node_connectivity[frame["edge_index"].to_numpy()]
            ]
            edges = sorted({edge_key(np.round(edge, 3)) for edge in endpoints})
    else:
        result = snap_gen_to_grid(
            geometry, like=like, window=arguments.get("window"),
            cellsize=cellsize,
        )
        edges = sorted(edge_key(edge) for edge in imod5_tests.wall_set(result))
    return {
        "name": name,
        "lines": [np.round(line, 6).tolist() for line in lines],
        "xnodes": x_nodes(grid).tolist(), "ynodes": y_nodes(grid).tolist(),
        "reference": output_edges(reference), "candidate": edges, "error": error,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--default-xugrid", action="store_true",
        help="Create the separate one-off raw xugrid comparison, not the repository report.",
    )
    options = parser.parse_args(argv)
    if not imod5_tests.IMOD5_EXE.is_file():
        raise FileNotFoundError(
            f"iMOD5 executable not found: {imod5_tests.IMOD5_EXE}. Set IMOD5_EXE to its path."
        )
    records = []
    with tempfile.TemporaryDirectory(prefix="grid-snapping-") as work:
        work = Path(work)
        for name in imod5_tests.GEOMETRY_CASES:
            case = imod5_tests.CASES[name]
            arguments = {"window": (0., 0., 4., 4.), "cellsize": 1.}
            if case.grid != "window":
                idf_dir = work / (name + "-idf")
                idf_dir.mkdir()
                top, _ = imod5_tests.write_case_idfs(idf_dir, case)
                arguments = {"idffile": top}
                if case.grid == "idffile-window":
                    arguments["window"] = imod5_tests.IDF_WINDOW
                    arguments["cellsize"] = 1.
            records.append(measure(
                name, case.lines, arguments, work / name, options.default_xugrid
            ))
        nonuniform = imod.util.empty_2d(
            dx=np.array(imod5_tests.NONUNIFORM_DX), xmin=0., xmax=sum(imod5_tests.NONUNIFORM_DX),
            dy=-np.array(imod5_tests.NONUNIFORM_DY[::-1]), ymin=0., ymax=sum(imod5_tests.NONUNIFORM_DY),
        )
        idf = imod5_tests._save_idf(work / "nonuniform-idf", "grid", nonuniform * 0. + 1.)
        for name, lines in imod5_tests.NONUNIFORM_CASES.items():
            records.append(measure(
                "nonuniform-" + name, lines, {"idffile": idf},
                work / ("nonuniform-" + name), options.default_xugrid,
            ))
        random = imod5_tests.make_random_lines(12, 12, 500, np.random.default_rng(42))
        perturbed_rng = np.random.default_rng(43)
        perturbed = [
            (np.asarray(line) + perturbed_rng.uniform(-0.02, 0.02, (len(line), 2))).tolist()
            for line in random
        ]
        for name, lines in (
            ("random-ties", random), ("random-perturbed", perturbed),
            ("random-feature-97", [random[97]]),
        ):
            records.append(measure(
                name, lines, {"window": (0., 0., 12., 12.), "cellsize": 1.},
                work / name, options.default_xugrid,
            ))
        for seed in (42, 7, 2026):
            name = f"large-meanders-{seed}"
            lines = imod5_tests.in_coordinates(
                imod5_tests.make_meandering_lines(1040, 1320, np.random.default_rng(seed)), "large"
            )
            window, cellsize = imod5_tests.grid_window("large")
            records.append(measure(
                name, lines, {"window": window, "cellsize": cellsize},
                work / name, options.default_xugrid,
            ))
    metadata = {
        "reference": "Actual iMOD5 executable (GENSNAPTOGRID)",
        "executable": str(imod5_tests.IMOD5_EXE),
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in ("xugrid", "imod")},
    }
    destination = HERE / "default_xugrid" if options.default_xugrid else HERE
    label = "Default xugrid" if options.default_xugrid else "Repository implementation"
    write_report(records, destination, label, metadata)
    print(f"Rebuilt {len(records)} comparisons: {destination / 'known_differences.html'}")


if __name__ == "__main__":
    main()
