import configparser
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    ValidationInfo,
    field_validator,
    model_validator,
)

imod_logger = logging.getLogger("imod")

_GENSNAPTOGRID = "GENSNAPTOGRID"
_GENPUZZLE = "GENPUZZLE"
_MKWELLIPF = "MKWELLIPF"
_DEFAULT_TOLERANCE = 0.001

IssMode = Annotated[
    # normalize input to lowercase and strip whitespace
    Literal["steady-state", "transient"],
    BeforeValidator(lambda v: v.strip().lower() if isinstance(v, str) else v),  
]
"""IssMode represents the mode of the simulation: either 'steady-state' or 'transient'."""

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "10": logging.DEBUG,
    "20": logging.INFO,
    "30": logging.WARNING,
    "40": logging.ERROR,
    "50": logging.CRITICAL,
}


def _normalise_mapping(data: Any) -> Any:
    """Make keywords case-insensitive and treat blank values as 'not provided'."""
    if not isinstance(data, dict):
        return data
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue  # blank -> field default applies
        out[key.strip().lower() if isinstance(key, str) else key] = value
    return out


def _resolve_against_base_dir(v: Any, info: ValidationInfo) -> Any:
    """Resolve relative paths against context["base_dir"], if given."""
    base_dir = (info.context or {}).get("base_dir")
    if base_dir is None or not isinstance(v, (str, Path)):
        return v
    return (Path(base_dir) / v).resolve()  # absolute paths are kept as-is


def _parse_window(v: Any) -> Any:
    if isinstance(v, str):
        bbox = tuple(float(part) for part in v.split(","))
        if not len(bbox) == 4:
            raise ValueError(
                f"WINDOW must have exactly four comma-separated values: xmin,ymin,xmax,ymax. Got: {v!r}"
            )
        v = bbox
    return v


def _check_window_bounds(bbox: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = bbox
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            "WINDOW must be xmin,ymin,xmax,ymax with xmin < xmax and ymin < ymax."
        )


def _parse_log_level(v: Any) -> int:
    msg = f"LOGLEVEL must be one of {list(_LOG_LEVELS.keys())}. Got: {v!r}"
    if isinstance(v, str):
        if v.lower() not in _LOG_LEVELS:
            raise ValueError(msg)
        return _LOG_LEVELS[v.lower()]
    elif isinstance(v, int):
        if v not in _LOG_LEVELS.values():
            raise ValueError(msg)
        return v
    else:
        raise ValueError(msg)  # noqa: TRY004


def _numbered_keys(prefix: str, count: int) -> list[str]:
    """The keys prefix1 to prefixN."""
    return [f"{prefix}{i}" for i in range(1, count + 1)]


def _numbered_values(
    data: dict[str, Any], prefix: str, count: int, required: bool = True
) -> list[Any]:
    """The values of the keys prefix1 to prefixN, in order. Missing keys are skipped unless required."""
    keys = _numbered_keys(prefix, count)
    missing = [key for key in keys if key not in data]
    if required and missing:
        raise ValueError(
            f"{missing[0].upper()} is missing, {prefix.upper()}1 to {prefix.upper()}{count} are required."
        )
    return [data[key] for key in keys if key in data]


def _number_or_path(v: Any, info: ValidationInfo) -> Any:
    """A constant for a layer, or else the path of a file."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return _resolve_against_base_dir(v, info)


class ConvertSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    prjfile_path: FilePath = Field(alias="prjfile_in")
    out_dir: Path = Field(alias="output_folder")
    overwrite: bool = Field(False, alias="overwrite")
    iss: IssMode = Field(alias="iss")
    model_name: str = Field("imported", alias="modelname")
    start_date: date = Field(date(1970, 1, 1), alias="sdate")
    end_date: date = Field(date(1970, 1, 2), alias="edate")
    interval: str = Field("D", alias="interval")
    bin_dir: DirectoryPath | None = Field(None, alias="coupler_dir")
    msw_dbase: Path | None = Field(None, alias="msw_dbase")  
    dump_netcdf: bool = Field(False, alias="dump_netcdf")
    # existence checked in validate_msw
    bbox: tuple[float, float, float, float] | None = Field(None, alias="window")
    cellsize: float | None = Field(None, gt=0, alias="cellsize")
    log_level: int = Field(logging.INFO, alias="loglevel")

    @property
    def is_steady_state(self) -> bool:
        return self.iss == "steady-state"

    @model_validator(mode="before")
    @classmethod
    def _normalise_input(cls, data: Any) -> Any:
        """Make keywords case-insensitive and treat blank values as 'not provided'."""
        return _normalise_mapping(data)

    @field_validator("prjfile_path", "out_dir", "bin_dir", "msw_dbase", mode="before")
    @classmethod
    def _resolve_relative(cls, v: Any, info: ValidationInfo) -> Any:
        """Resolve relative paths against context["base_dir"], if given."""
        return _resolve_against_base_dir(v, info)

    @field_validator("interval")
    @classmethod
    def _check_interval(cls, v: str) -> str:
        """Fail early on anything pandas does not accept as an offset alias."""
        try:
            pd.tseries.frequencies.to_offset(v)
        except ValueError as e:
            raise ValueError(
                f"INTERVAL {v!r} is not a valid pandas offset alias: {e}"
            ) from e
        return v

    @field_validator("bbox", mode="before")
    @classmethod
    def _parse_bbox(cls, v: Any) -> Any:
        return _parse_window(v)

    @model_validator(mode="after")
    def _check_consistency(self) -> "ConvertSettings":
        if (self.bbox is None) != (self.cellsize is None):
            raise ValueError(
                "Both bbox (WINDOW) and cellsize must be provided together to create target grid."
            )
        if self.bbox is not None:
            _check_window_bounds(self.bbox)
        if not self.is_steady_state and self.end_date < self.start_date:
            raise ValueError("EDATE must not be before SDATE.")
        if self.out_dir.is_dir() and any(self.out_dir.iterdir()) and not self.overwrite:
            raise ValueError(
                f"Output folder is not empty: {self.out_dir}, "
                "please either empty it or set 'overwrite' to True."
            )
        return self

    @field_validator("log_level", mode="before")
    @classmethod
    def _log_level_to_int(cls, v: str) -> int:
        return _parse_log_level(v)

    def validate_msw(self) -> None:
        """Extra checks, only needed when creating a MetaSwap model."""
        if self.msw_dbase is None:
            raise ValueError(
                "MetaSwap database path must be provided to create MetaSwap model."
            )
        if self.bin_dir is None:
            raise ValueError(
                "Coupler binaries directory must be provided to create MetaSwap model."
            )
        if not self.msw_dbase.is_dir():
            raise FileNotFoundError(f"MetaSwap database not found at {self.msw_dbase}")

    def simulation_times(self) -> list[pd.Timestamp]:
        return pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.interval
        ).tolist()


class GenBaseSettings(BaseModel):
    """Base settings for GEN file processing functions."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    genfile_in: FilePath = Field(alias="genfile_in")
    genfile_out: Path = Field(alias="genfile_out")
    function: str = Field(alias="function")
    overwrite: bool = Field(False, alias="overwrite")
    log_level: int = Field(logging.INFO, alias="loglevel")

    @model_validator(mode="before")
    @classmethod
    def _normalise_input(cls, data: Any) -> Any:
        """Make keywords case-insensitive and treat blank values as 'not provided'."""
        return _normalise_mapping(data)

    @field_validator("genfile_in", "genfile_out", mode="before")
    @classmethod
    def _resolve_relative(cls, v: Any, info: ValidationInfo) -> Any:
        """Resolve relative paths against context["base_dir"], if given."""
        return _resolve_against_base_dir(v, info)

    @field_validator("log_level", mode="before")
    @classmethod
    def _log_level_to_int(cls, v: str) -> int:
        return _parse_log_level(v)

    @field_validator("function", mode="before")
    @classmethod
    def _check_function(cls, v: Any) -> str:
        raise NotImplementedError(
            "Function validation is not implemented yet."
        )

class GenSnapToGridSettings(GenBaseSettings):
    """
    ConvertSettings for the GENSNAPTOGRID function (iMOD5 manual, section 8.4.1): snap
    the lines in a GEN file to the cell edges of a grid.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    idffile: FilePath | None = Field(None, alias="idffile")
    bbox: tuple[float, float, float, float] | None = Field(None, alias="window")
    cellsize: float | None = Field(None, gt=0, alias="cellsize")
    max_snap_distance: float | None = Field(None, gt=0, alias="max_snap_distance")
    i3d: int = Field(0, alias="i3d")

    @field_validator("function", mode="before")
    @classmethod
    def _check_function(cls, v: Any) -> str:
        if isinstance(v, str) and v.strip().upper() == _GENSNAPTOGRID:
            return _GENSNAPTOGRID
        raise ValueError(f"FUNCTION must be {_GENSNAPTOGRID}. Got: {v!r}")

    @field_validator("bbox", mode="before")
    @classmethod
    def _parse_bbox(cls, v: Any) -> Any:
        return _parse_window(v)

    @field_validator("i3d", mode="before")
    @classmethod
    def _check_i3d(cls, v: Any) -> int:
        i3d = int(v)
        if i3d == 1:
            raise ValueError(
                "I3D=1 (3D GEN output) is not supported yet, only I3D=0 (2D GEN)."
            )
        if i3d != 0:
            raise ValueError(f"I3D must be 0. Got: {v!r}")
        return i3d

    @model_validator(mode="after")
    def _check_consistency(self) -> "GenSnapToGridSettings":
        if self.idffile is None:
            if self.bbox is None or self.cellsize is None:
                raise ValueError(
                    "Provide either IDFFILE, or both WINDOW and CELLSIZE, "
                    "to define the grid to snap to."
                )
        elif self.cellsize is not None:
            raise ValueError(
                "CELLSIZE cannot be combined with IDFFILE: the cell size of the "
                "IDF defines the grid. Use WINDOW to reduce its extent."
            )
        if self.bbox is not None:
            _check_window_bounds(self.bbox)
        if self.genfile_out.exists() and not self.overwrite:
            raise ValueError(
                f"Output GEN file already exists: {self.genfile_out}, "
                "please either remove it or set 'overwrite' to True."
            )
        return self


class GenPuzzleSettings(GenBaseSettings):
    """
    Settings for the GENPUZZLE function: connect loose line segments in a GEN file
    into continuous segments, following the GENPUZZLE function in iMOD5.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    tolerance: float = Field(_DEFAULT_TOLERANCE, gt=0, alias="tolerance")

    @field_validator("function", mode="before")
    @classmethod
    def _check_function(cls, v: Any) -> str:
        if isinstance(v, str) and v.strip().upper() == _GENPUZZLE:
            return _GENPUZZLE
        raise ValueError(f"FUNCTION must be {_GENPUZZLE}. Got: {v!r}")

    @model_validator(mode="after")
    def _check_consistency(self) -> "GenPuzzleSettings":
        if self.genfile_out.exists() and not self.overwrite:
            raise ValueError(
                f"Output GEN file already exists: {self.genfile_out}, "
                "please either remove it or set 'overwrite' to True."
            )
        return self


class MkWellIpfSettings(BaseModel):
    """
    Settings for the MKWELLIPF function: spread the pumping rate of wells over the model layers,
    following the MKWELLIPF function in iMOD5, see the iMOD5 manual for the meaning of the
    keywords. Column numbers start at 1. Layers must touch, as in MODFLOW 6. In an ini file the
    files of the layers and the IPFs are given with numbered keys (``TOPIDF1``, ``IPF1``, ...).
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    function: str = Field(_MKWELLIPF, alias="function")
    ipf_files: list[FilePath] = Field(min_length=1)
    top_files: list[FilePath] | None = None
    bottom_files: list[FilePath] | None = None
    khv_files: list[FilePath | float] | None = None
    kd_files: list[FilePath | float] | None = None
    iss: Literal[0, 1] = 0
    sdate: date | None = None
    edate: date | None = None
    x_col: int = Field(1, ge=1, alias="ixcol")
    y_col: int = Field(2, ge=1, alias="iycol")
    q_col: int = Field(1, ge=1, alias="iqcol")
    top_col: int = Field(4, ge=1, alias="itcol")
    bottom_col: int = Field(5, ge=1, alias="ibcol")
    minkht: float = Field(0.0, ge=0)
    minkd: float = Field(0.0, ge=0)
    fnodata: float = -99999.0
    hnodata: float = 0.0
    overwrite: bool = False
    output_folder: Path | None = None
    log_level: int = Field(logging.INFO, alias="loglevel")

    @model_validator(mode="before")
    @classmethod
    def _collect_numbered_keys(cls, data: Any, info: ValidationInfo) -> Any:
        """
        Make keywords case-insensitive, and collect the numbered keywords of an ini file
        (TOPIDF1.., IPF1..) into the lists of files.
        """
        data = _normalise_mapping(data)
        if not isinstance(data, dict):
            return data
        if int(data.get("imidf", 0)) != 0:
            raise ValueError("IMIDF is not supported yet, only IMIDF=0.")
        # The number of IPFs and layers are given as keywords, use these to find the files
        collected: dict[str, list[Any]] = {}
        consumed = {"nipf", "nlay", "imidf"}
        if "ipf_files" not in data:
            if "nipf" not in data:
                raise ValueError("NIPF and IPF1 to IPFN are required.")
            nipf = int(data["nipf"])
            collected["ipf_files"] = [
                _resolve_against_base_dir(path, info)
                for path in _numbered_values(data, "ipf", nipf)
            ]
            consumed.update(_numbered_keys("ipf", nipf))
        nlay = int(data.get("nlay", 0))
        for key, prefix in (("top_files", "topidf"), ("bottom_files", "botidf")):
            if nlay > 0:
                collected[key] = [
                    _resolve_against_base_dir(path, info)
                    for path in _numbered_values(data, prefix, nlay)
                ]
                consumed.update(_numbered_keys(prefix, nlay))
        for key, prefix in (("khv_files", "khvidf"), ("kd_files", "kdvidf")):
            values = _numbered_values(data, prefix, nlay, required=False)
            if values:
                collected[key] = [_number_or_path(value, info) for value in values]
            consumed.update(_numbered_keys(prefix, nlay))
        return {**{k: v for k, v in data.items() if k not in consumed}, **collected}

    @field_validator("function", mode="before")
    @classmethod
    def _check_function(cls, v: Any) -> str:
        if isinstance(v, str) and v.strip().upper() == _MKWELLIPF:
            return _MKWELLIPF
        raise ValueError(f"FUNCTION must be {_MKWELLIPF}. Got: {v!r}")

    @field_validator("iss", mode="before")
    @classmethod
    def _iss_to_int(cls, v: Any) -> Any:
        return int(v) if isinstance(v, str) else v

    @field_validator("sdate", "edate", mode="before")
    @classmethod
    def _parse_date(cls, v: Any) -> Any:
        """Dates as in iMOD5 (yyyymmdd) or ISO (yyyy-mm-dd)."""
        if isinstance(v, (str, int)) and len(str(v).strip()) == 8 and str(v).strip().isdigit():
            return datetime.strptime(str(v).strip(), "%Y%m%d").date()
        return v

    @field_validator("output_folder", mode="before")
    @classmethod
    def _resolve_output_folder(cls, v: Any, info: ValidationInfo) -> Any:
        """Resolve a relative path against context["base_dir"], if given."""
        return _resolve_against_base_dir(v, info)

    @field_validator("log_level", mode="before")
    @classmethod
    def _log_level_to_int(cls, v: Any) -> int:
        return _parse_log_level(v)

    @model_validator(mode="after")
    def _check_consistency(self) -> "MkWellIpfSettings":
        nlayer = len(self.top_files or [])
        if len(self.bottom_files or []) != nlayer:
            raise ValueError("TOPIDF and BOTIDF must be given for the same number of layers.")
        k_files = [files for files in (self.khv_files, self.kd_files) if files is not None]
        if len(k_files) > 1:
            raise ValueError(
                "Give either KHVIDF or KDVIDF files, not both. A mix between layers is not supported."
            )
        if k_files and len(k_files[0]) != nlayer:
            raise ValueError("KHVIDF or KDVIDF must be given for every layer of TOPIDF and BOTIDF.")
        if self.iss == 1 and (self.sdate is None or self.edate is None):
            raise ValueError("SDATE and EDATE are required when ISS=1.")
        if self.sdate and self.edate and self.edate < self.sdate:
            raise ValueError("EDATE must not be before SDATE.")
        names = [path.stem.lower() for path in self.ipf_files]
        if self.output_folder is not None and len(set(names)) != len(names):
            raise ValueError("The IPF files must have different names when OUTPUT_FOLDER is given.")
        return self


def _read_ini[T: BaseModel](model: type[T], inifile: Path | str) -> T:
    imod_logger.info(f"Reading settings from {inifile}")
    inifile = Path(inifile)
    config = configparser.ConfigParser(allow_unnamed_section=True, interpolation=None)
    with inifile.open(encoding="utf-8") as f:
        config.read_file(f)
    return model.model_validate(
        dict(config[configparser.UNNAMED_SECTION]),
        context={"base_dir": inifile.resolve().parent},
    )


def read_convert_settings(inifile: Path | str) -> ConvertSettings:
    return _read_ini(ConvertSettings, inifile)


def read_gen_snap_to_grid_settings(inifile: Path | str) -> GenSnapToGridSettings:
    return _read_ini(GenSnapToGridSettings, inifile)


def read_genpuzzle_settings(inifile: Path | str) -> GenPuzzleSettings:
    return _read_ini(GenPuzzleSettings, inifile)


def read_mkwellipf_settings(inifile: Path | str) -> MkWellIpfSettings:
    return _read_ini(MkWellIpfSettings, inifile)
