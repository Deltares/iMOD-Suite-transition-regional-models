# NHI sprint

This repository features created during the sprints for the NHI. Currently supported are:

1. Convert an iMOD5 database to a coupled MODFLOW6-MetaSWAP simulation.

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
pixi run convert conversion_Peelvenen.ini``
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
| `COUPLER_DIR` | Yes | None | Directory path (string) | Path to the folder containing coupler binaries/executables. |
| `MSW_DBASE` | Yes | None | Directory path (string) | Path to the MetaSWAP database directory used during conversion. |
| `MODELNAME` | No | `imported` | String | Short model identifier used for naming outputs and metadata. |
| `OUTPUT_FOLDER` | Yes | None | Directory path (string) | Destination folder where converted files are written. |
| `SDATE` | Yes | None | Date string (`YYYY-MM-DD`) | Start date of the simulation period to convert. |
| `EDATE` | Yes | None | Date string (`YYYY-MM-DD`) | End date of the simulation period to convert. |
| `INTERVAL` | No | `D` | String | Time step interval code (for example `D` for daily). [Anything accepted by pandas is supported](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) |
| `WINDOW` | No* | `None` | Comma-separated numbers | Spatial extent as `xmin,ymin,xmax,ymax` in model coordinates. |
| `CELLSIZE` | No* | `None` | Number (integer/float) | Target horizontal grid cell size (typically in meters). |

\* `WINDOW` and `CELLSIZE` are optional together. If one is provided, the other must also be provided.
