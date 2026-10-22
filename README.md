# NHI sprint

This repository features created during the sprints for the NHI. Currently supported are:

1. Convert an iMOD5 database to a coupled MODFLOW6-MetaSWAP simulation.
2. Backwards compatible versions of iMOD5 batch functions (currently
   `GENSNAPTOGRID`), see [GEN-batch functions](#gen-batch-functions).

##  Install

### With pixi (recommended, requires internet connection)

Note that this requires internet connection.

This repository uses pixi to manage and install python environments. 
[Read how to install pixi here](https://pixi.prefix.dev/latest/installation/)
As for now, installation requires internet access. 

#### Some background info

The ``pixi.lock`` contains the exact builds of each package in the python
environment and where to get it. It therefore ensures that everyone using this
repository will install exactly the same versions of python packages.
Furthermore, pixi installs very fast. It will install the python packages in a
``.pixi`` folder. You can delete this after you are done with this project: The
python environment can be simply reproduced based on the ``pixi.toml`` and
``pixi.lock`` file.

### Using the environment.ps1 (offline, note this install is experimental)

Open powershell, call:

```powershell
.\environment.ps1
```

This will generate the python environment and an ``activate.bat`` script to
activate this environment.

## Convert iMOD5 model

If you installed with pixi:

```powershell
pixi run convert <name of your ini file>
```

For example:

```powershell
pixi run convert conversion_Peelvenen.ini
```

This will install the pixi environment automatically and consequently run the
python script with the ini file you provided.

If you followed the offline installation instructions, the commands are slightly
different:

```powershell
.\activate.bat
python convert <name of your ini file>
```

### Configuration

Arguments supported:

| Argument | Required | Default/Fallback | Datatype | Brief explanation |
| --- | --- | --- | --- | --- |
| `PRJFILE_IN` | Yes | None | Path (string) | Path to the iMOD5 `.PRJ` input project file. |
| `MODELNAME` | No | `imported` | String | Short model identifier used for naming outputs and metadata. |
| `OUTPUT_FOLDER` | Yes | None | Directory path (string) | Destination folder where converted files are written. |
| `OVERWRITE` | No | `false` | Boolean | Allow overwriting data in an existing output folder. |
| `ISS` | Yes | None | String | Simulation type: `steady-state` or `transient`. |
| `SDATE` | No | "1970-01-01" | Date string (`YYYY-MM-DD`) | Start date of the simulation period to convert. |
| `EDATE` | No | "1970-01-01" | Date string (`YYYY-MM-DD`) | End date of the simulation period to convert. |
| `INTERVAL` | No | `D` | String | Time step interval code (for example `D` for daily). [Anything accepted by pandas is supported](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) |
| `COUPLER_DIR` | No* | None | Directory path (string) | Path to the folder containing coupler binaries/executables. |
| `MSW_DBASE` | No* | None | Directory path (string) | Path to the MetaSWAP database directory used during conversion. |
| `WINDOW` | No** | `None` | Comma-separated numbers | Spatial extent as `xmin,ymin,xmax,ymax` in model coordinates. |
| `CELLSIZE` | No** | `None` | Number (integer/float) | Target horizontal grid cell size (typically in meters). |
| `LOGLEVEL` | NO | `INFO` | Number (integer) or string. | The level of verbosity in the log file, choose from [`DEBUG`/`10`, `INFO`/`20`, `WARNING`/`30`, `ERROR`/`40`, `CRITICAL`/`50`] |

\* Only required when there is CAP entry in the projectfile for MetaSWAP.

\*\* `WINDOW` and `CELLSIZE` are optional together. If one is provided, the other must also be provided.

## GEN-batch functions

This repository provides backwards compatible versions of some Batch functions of IMOD5.

| Imod5 function | pixi task |
| --- | --- |
| GENSNAPTOGRID | gensnaptogrid |

The way to run these batch functions is very similar to the IMOD5 implementation and the conversion task in this repo.

If you installed with pixi:

```powershell
pixi run <task name> <path of your ini file>
```

### GENSNAPTOGRID

Rasterizes the lines in a GEN file onto the cell edges of a grid, so you can see
how the lines will be applied in a model for a given cell size. This is the
backwards compatible version of the iMOD5 function `GENSNAPTOGRID` (iMOD5 user
manual, section 8.4.1). Only the 2D case (`I3D=0`) is supported.

```powershell
pixi run gensnaptogrid snap_gen.ini
```

Example ini file, snapping to a grid defined by a window and a cell size:

```ini
FUNCTION=GENSNAPTOGRID
GENFILE_IN=data\faults.gen
GENFILE_OUT=data\faults_snapped.gen
WINDOW=100000.0,400000.0,200000.0,425000.0
CELLSIZE=100.0
```

Or, snapping to the grid of an IDF:

```ini
FUNCTION=GENSNAPTOGRID
GENFILE_IN=data\faults.gen
GENFILE_OUT=data\faults_snapped.gen
IDFFILE=data\bnd.idf
```

Relative paths are resolved against the folder of the ini file. Keywords are
not case sensitive.

Arguments supported:

| Argument | Required | Default/Fallback | Datatype | Brief explanation |
| --- | --- | --- | --- | --- |
| `FUNCTION` | Yes | None | String | Must be `GENSNAPTOGRID`. |
| `GENFILE_IN` | Yes | None | Path (string) | Input GEN file (ASCII or binary), containing 2D lines. |
| `GENFILE_OUT` | Yes | None | Path (string) | Output GEN file to create (binary). |
| `IDFFILE` | No* | None | Path (string) | IDF whose grid is used to snap the lines to. |
| `WINDOW` | No* | `None` | Comma-separated numbers | Spatial extent as `xmin,ymin,xmax,ymax`. Together with `IDFFILE`, only the cells with their centre inside the window are used. |
| `CELLSIZE` | No* | `None` | Number (integer/float) | Cell size of the grid to snap to, in the same units as the GEN file. Cannot be combined with `IDFFILE`. |
| `MAX_SNAP_DISTANCE` | No | not used | Number (greater than zero) | Not part of iMOD5. Moves line vertices within this distance onto the nearest grid node before snapping. By default vertices are not moved, as in iMOD5; setting this changes which edges are selected. |
| `I3D` | No | `0` | Integer | Only `0` (2D GEN) is supported. `1` (3D GEN) is rejected. |
| `OVERWRITE` | No | `false` | Boolean | Allow overwriting an existing `GENFILE_OUT`. |
| `LOGLEVEL` | No | `INFO` | Number (integer) or string. | The level of verbosity in the log file, choose from [`DEBUG`/`10`, `INFO`/`20`, `WARNING`/`30`, `ERROR`/`40`, `CRITICAL`/`50`] |

\* Provide either `IDFFILE`, or both `WINDOW` and `CELLSIZE`.

#### How lines are snapped

The result was compared against iMOD5 5.6.1 on synthetic GEN files (see
[Testing against iMOD5](#testing-against-imod5)):

- A line selects a cell edge when it crosses the straight segment between the
  centres of the two cells that share that edge. Vertices are not moved.
- The edge of the cell that contains the last vertex of a line is also selected,
  even if the line stops before the centre of that cell. This is what iMOD5 does.
  The first vertex is not extended, so reversing a line changes the result.
- Only edges between two cells are selected: edges on the boundary of the grid
  are never snapped to. A line along the boundary, a line shorter than a cell, or
  a line outside the grid gives no output. If no line is snapped at all, no
  output file is written and the function raises an error.
- Lines are clipped to the grid. Every input feature keeps its own output line
  and attributes, also if two lines snap to the same edge.
- If a line leaves and re-enters the grid, the parts are written as separate
  features with the attributes repeated, as a GEN file has no multi-part lines.
- GEN files do not store a coordinate system, so the lines and the grid must use
  the same one.

Known difference: lines that pass exactly through cell centres (for example a
45 degree line from centre to centre) are ties. iMOD5 resolves these differently
from this implementation, and the result of this implementation can depend on
floating point rounding.

#### Testing against iMOD5

`tests\test_snapping_imod5.py` writes synthetic GEN files, snaps them with both
this implementation and the iMOD5 executable, and compares the edges they select.
The tests are skipped when the executable cannot be found. By default it is
looked for at `C:\Development\bin\iMOD5\iMOD 5.6.1\iMOD_V5_6_1.exe`; set the
environment variable `IMOD5_EXE` to use another location:

```powershell
$env:IMOD5_EXE = "C:\path\to\iMOD_V5_6_1.exe"
pixi run pytest tests\test_snapping_imod5.py
```

iMOD5 receives the settings as an `.ini` file. For every case a folder
`_scratch\imod5_snap\<grid>_<case>\` is written with the input, both outputs, the
`.ini` file, a `diff.txt` and a `comparison.png` (green: input, red: this
implementation, blue dashed: iMOD5), so differences can be inspected.
