# Grid snapping comparisons

Rebuild the comparison between **actual iMOD5 executable output** and this
repository's public snapping implementation, including the Markdown report,
self-contained HTML gallery, SVG figures and measured JSON data:

```powershell
pixi run compare-grid-snapping
```

Run from the repository root. The task requires iMOD5 5.6.1 and the existing
pixi environment. If needed, set its executable before running:

```powershell
$env:IMOD5_EXE = 'C:\path\to\iMOD_V5_6_1.exe'
pixi run compare-grid-snapping
```

Inputs and executable outputs are rebuilt in temporary storage on every run;
no scratch scripts, cached outputs or existing documentation files are needed.
Only the measured JSON, reports and a figure for every case that differs from
the executable are retained here.
Missing executables and failed runs raise errors rather than silently reusing
historical data. The cases reuse the executable integration tests.

- [Repository comparison](known_differences.md)
- [Repository HTML gallery](known_differences.html)
- [One-off default-xugrid comparison](default_xugrid/known_differences.md)
- [One-off default-xugrid HTML gallery](default_xugrid/known_differences.html)

The separate default-xugrid snapshot was generated once with:

```powershell
pixi run python -m docs.grid_snapping.compare --default-xugrid
```

It uses `max_snap_distance=0.0` and xugrid's default tolerance, without the
repository's outside guard. It is **not** rebuilt by the normal pixi task.
Library failures appear as explicit errors in that report.

Counts are combined unique geometric walls, not per-feature assignments.
Wall endpoints are rounded to three decimal places, using the same convention
as the executable integration tests; input GEN coordinates use six decimals.
Dense random samples may cover identical combined walls while assigning them
to different features. These synthetic 2D comparisons do not assess attributes,
3D output, groundwater heads or flows.