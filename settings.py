import configparser
import logging
from datetime import date
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

IssMode = Annotated[
    Literal["steady-state", "transient"],
    BeforeValidator(lambda v: v.strip().lower() if isinstance(v, str) else v),
]


class Settings(BaseModel):
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
    msw_dbase: Path | None = Field(None, alias="msw_dbase")  # existence checked in validate_msw
    bbox: tuple[float, float, float, float] | None = Field(None, alias="window")
    cellsize: float | None = Field(None, gt=0, alias="cellsize")

    @property
    def is_steady_state(self) -> bool:
        return self.iss == "steady-state"

    @model_validator(mode="before")
    @classmethod
    def _normalise_input(cls, data: Any) -> Any:
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

    @field_validator("prjfile_path", "out_dir", "bin_dir", "msw_dbase", mode="before")
    @classmethod
    def _resolve_relative(cls, v: Any, info: ValidationInfo) -> Any:
        """Resolve relative paths against context["base_dir"], if given."""
        base_dir = (info.context or {}).get("base_dir")
        if base_dir is None or not isinstance(v, (str, Path)):
            return v
        return (Path(base_dir) / v).resolve()  # absolute paths are kept as-is

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
        if isinstance(v, str):
            bbox = tuple(float(part) for part in v.split(","))
            if not len(bbox) == 4:
                raise ValueError(
                    f"WINDOW must have exactly four comma-separated values: xmin,ymin,xmax,ymax. Got: {v!r}"
                )
            v = bbox
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if (self.bbox is None) != (self.cellsize is None):
            raise ValueError(
                "Both bbox (WINDOW) and cellsize must be provided together to create target grid."
            )
        if self.bbox is not None:
            xmin, ymin, xmax, ymax = self.bbox
            if xmin >= xmax or ymin >= ymax:
                raise ValueError(
                    "WINDOW must be xmin,ymin,xmax,ymax with xmin < xmax and ymin < ymax."
                )
        if not self.is_steady_state and self.end_date < self.start_date:
            raise ValueError("EDATE must not be before SDATE.")
        if self.out_dir.is_dir() and any(self.out_dir.iterdir()) and not self.overwrite:
            raise ValueError(
                f"Output folder is not empty: {self.out_dir}, "
                "please either empty it or set 'overwrite' to True."
            )
        return self

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

def read_settings(inifile: Path | str) -> Settings:
    imod_logger.info(f"Reading settings from {inifile}")
    inifile = Path(inifile)
    config = configparser.ConfigParser(allow_unnamed_section=True, interpolation=None)
    with inifile.open(encoding="utf-8") as f:
        config.read_file(f)
    return Settings.model_validate(
        dict(config[configparser.UNNAMED_SECTION]),
        context={"base_dir": inifile.resolve().parent},
    )
