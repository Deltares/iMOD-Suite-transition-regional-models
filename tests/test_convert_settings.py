import configparser
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.settings import ConvertSettings, read_convert_settings

PEELVENEN_INI = """\
PRJFILE_IN=../Peelvenen/prjfiles/Peelvenen_relative_paths.PRJ
COUPLER_DIR=../imod_coupler_windows
MSW_DBASE=../data/MSW_dbase
MODELNAME=peelvenen
OUTPUT_FOLDER=../Peelvenen/conversion_output
ISS=transient
SDATE=2011-01-01
EDATE=2011-01-07
INTERVAL=D
WINDOW=175000,364000,201000,397000
CELLSIZE=100"""

PEELVENEN_SS_INI = """\
PRJFILE_IN=../Peelvenen/prjfiles/Peelvenen_relative_paths_steady_state.PRJ
COUPLER_DIR=../imod_coupler_windows
MSW_DBASE=../data/MSW_dbase
MODELNAME=peelvenen
OUTPUT_FOLDER=../Peelvenen/conversion_output_ss
ISS=steady-state
WINDOW=175000,364000,201000,397000
CELLSIZE=100"""

# Has no ISS, which is a required keyword.
MODEL_CP_INI = """\
PRJFILE_IN=../Model_CP/RUNFILES/BASIS1/BASIS1_TA-MF6-UNCONF-50/IBR30_BASIS1_TA-MF6-UNCONF-50-adapted.PRJ
COUPLER_DIR=../bin
MSW_DBASE=../data/MSW_dbase
OUTPUT_FOLDER=../Model_CP/conversion_output
SDATE=2011-01-01
EDATE=2011-01-07
INTERVAL=D"""


def write_ini(path: Path, entries: dict[str, str]) -> Path:
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in entries.items()), encoding="utf-8"
    )
    return path


def make_dir_tree(ini_dir: Path, files: list[str], dirs: list[str]) -> None:
    """Create relative targets as seen from `ini_dir`."""
    for rel in files:
        target = ini_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    for rel in dirs:
        (ini_dir / rel).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def raw(tmp_path: Path) -> dict[str, str]:
    """Minimal valid settings with real paths on disk."""
    prj = tmp_path / "model.PRJ"
    prj.touch()
    (tmp_path / "bin").mkdir()
    (tmp_path / "msw").mkdir()
    return {
        "PRJFILE_IN": str(prj),
        "OUTPUT_FOLDER": str(tmp_path / "out"),
        "ISS": "transient",
    }


@pytest.fixture
def ini_dir(tmp_path: Path) -> Path:
    """Directory holding the ini file; relative paths resolve against it."""
    d = tmp_path / "inifiles"
    d.mkdir()
    return d


# example ini files 
def test_peelvenen_transient_example(ini_dir: Path) -> None:
    make_dir_tree(
        ini_dir,
        files=["../Peelvenen/prjfiles/Peelvenen_relative_paths.PRJ"],
        dirs=["../imod_coupler_windows", "../data/MSW_dbase"],
    )
    ini = ini_dir / "conversion_Peelvenen.ini"
    ini.write_text(PEELVENEN_INI)

    s = read_convert_settings(ini)

    root = ini_dir.parent.resolve()
    assert s.prjfile_path == root / "Peelvenen/prjfiles/Peelvenen_relative_paths.PRJ"
    assert s.out_dir == root / "Peelvenen/conversion_output"
    assert s.bin_dir == root / "imod_coupler_windows"
    assert s.msw_dbase == root / "data/MSW_dbase"
    assert s.model_name == "peelvenen"
    assert s.iss == "transient"
    assert not s.is_steady_state
    assert s.start_date == date(2011, 1, 1)
    assert s.end_date == date(2011, 1, 7)
    assert s.interval == "D"
    assert s.bbox == (175000.0, 364000.0, 201000.0, 397000.0)
    assert s.cellsize == 100.0
    assert s.overwrite is False
    s.validate_msw()


def test_peelvenen_steady_state_example_uses_defaults(ini_dir: Path) -> None:
    make_dir_tree(
        ini_dir,
        files=["../Peelvenen/prjfiles/Peelvenen_relative_paths_steady_state.PRJ"],
        dirs=["../imod_coupler_windows", "../data/MSW_dbase"],
    )
    ini = ini_dir / "conversion_Peelvenen_ss.ini"
    ini.write_text(PEELVENEN_SS_INI)

    s = read_convert_settings(ini)

    assert s.is_steady_state
    assert s.start_date == date(1970, 1, 1)
    assert s.end_date == date(1970, 1, 2)
    assert s.interval == "D"
    assert s.bbox == (175000.0, 364000.0, 201000.0, 397000.0)
    assert s.cellsize == 100.0


def test_model_cp_example_without_iss_raises(ini_dir: Path) -> None:
    make_dir_tree(
        ini_dir,
        files=[
            "../Model_CP/RUNFILES/BASIS1/BASIS1_TA-MF6-UNCONF-50/IBR30_BASIS1_TA-MF6-UNCONF-50-adapted.PRJ"
        ],
        dirs=["../bin", "../data/MSW_dbase"],
    )
    ini = ini_dir / "conversion_ModelCP.ini"
    ini.write_text(MODEL_CP_INI)

    with pytest.raises(ValidationError) as exc:
        read_convert_settings(ini)

    assert [e["loc"] for e in exc.value.errors()] == [("iss",)]
    assert exc.value.errors()[0]["type"] == "missing"

# path resolution relative to the ini file
def test_relative_paths_resolve_against_ini_location_not_cwd(
    tmp_path: Path, ini_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_dir_tree(ini_dir, files=["../data/model.PRJ"], dirs=["../bin", "../msw"])
    ini = write_ini(
        ini_dir / "c.ini",
        {
            "PRJFILE_IN": "../data/model.PRJ",
            "OUTPUT_FOLDER": "../out",
            "ISS": "steady-state",
            "COUPLER_DIR": "../bin",
            "MSW_DBASE": "../msw",
        },
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    s = read_convert_settings(ini)

    root = tmp_path.resolve()
    assert s.prjfile_path == root / "data" / "model.PRJ"
    assert s.out_dir == root / "out"
    assert s.bin_dir == root / "bin"
    assert s.msw_dbase == root / "msw"


def test_base_dir_context_resolves_relative_paths_in_direct_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_dir_tree(tmp_path, files=["data/model.PRJ"], dirs=["bin"])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    s = ConvertSettings.model_validate(
        {
            "prjfile_in": "data/model.PRJ",
            "output_folder": "out",
            "iss": "transient",
            "coupler_dir": "bin",
        },
        context={"base_dir": tmp_path},
    )

    root = tmp_path.resolve()
    assert s.prjfile_path == root / "data" / "model.PRJ"
    assert s.out_dir == root / "out"
    assert s.bin_dir == root / "bin"


def test_without_context_relative_paths_resolve_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_dir_tree(tmp_path, files=["model.PRJ"], dirs=[])
    monkeypatch.chdir(tmp_path)

    s = ConvertSettings.model_validate(
        {"prjfile_in": "model.PRJ", "output_folder": "out", "iss": "transient"}
    )

    assert s.prjfile_path == Path("model.PRJ")
    assert s.out_dir == Path("out")


def test_ini_given_as_relative_path_resolves_correctly(
    ini_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_dir_tree(ini_dir, files=["model.PRJ"], dirs=[])
    write_ini(
        ini_dir / "c.ini",
        {"PRJFILE_IN": "model.PRJ", "OUTPUT_FOLDER": "out", "ISS": "transient"},
    )
    monkeypatch.chdir(ini_dir.parent)

    s = read_convert_settings(Path("inifiles/c.ini"))

    assert s.prjfile_path == ini_dir.resolve() / "model.PRJ"


def test_absolute_paths_are_kept(tmp_path: Path, raw: dict[str, str]) -> None:
    ini = write_ini(tmp_path / "c.ini", raw)

    s = read_convert_settings(ini)

    assert s.prjfile_path == Path(raw["PRJFILE_IN"]).resolve()
    assert s.out_dir == Path(raw["OUTPUT_FOLDER"]).resolve()


def test_missing_relative_prjfile_raises(ini_dir: Path) -> None:
    ini = write_ini(
        ini_dir / "c.ini",
        {"PRJFILE_IN": "nope.PRJ", "OUTPUT_FOLDER": "out", "ISS": "transient"},
    )

    with pytest.raises(ValidationError) as exc:
        read_convert_settings(ini)

    assert exc.value.errors()[0]["loc"] == ("prjfile_in",)


def test_blank_path_value_is_not_turned_into_ini_directory(ini_dir: Path) -> None:
    make_dir_tree(ini_dir, files=["model.PRJ"], dirs=[])
    ini = write_ini(
        ini_dir / "c.ini",
        {
            "PRJFILE_IN": "model.PRJ",
            "OUTPUT_FOLDER": "out",
            "ISS": "transient",
            "COUPLER_DIR": "",
            "MSW_DBASE": " ",
        },
    )

    s = read_convert_settings(ini)

    assert s.bin_dir is None
    assert s.msw_dbase is None


def test_path_keywords_resolve_regardless_of_key_case(ini_dir: Path) -> None:
    make_dir_tree(ini_dir, files=["model.PRJ"], dirs=[])
    ini = write_ini(
        ini_dir / "c.ini",
        {"PrjFile_In": "model.PRJ", "Output_Folder": "out", "Iss": "transient"},
    )

    assert read_convert_settings(ini).prjfile_path == ini_dir.resolve() / "model.PRJ"


# keyword case-insensitivity
@pytest.mark.parametrize("transform", [str.lower, str.upper, str.title, str.swapcase])
def test_keywords_are_case_insensitive_via_read_settings(
    tmp_path: Path, raw: dict[str, str], transform: Callable[[str], str]
) -> None:
    ini = write_ini(tmp_path / "c.ini", {transform(k): v for k, v in raw.items()})

    s = read_convert_settings(ini)

    assert s.iss == "transient"
    assert s.out_dir == Path(raw["OUTPUT_FOLDER"])


def test_keywords_are_case_insensitive_without_configparser(
    raw: dict[str, str],
) -> None:
    # Direct dict input bypasses configparser's own lowercasing.
    mixed = {
        "PrjFile_In": raw["PRJFILE_IN"],
        "OutPut_Folder": raw["OUTPUT_FOLDER"],
        "iSs": "transient",
    }

    s = ConvertSettings.model_validate(mixed)

    assert s.iss == "transient"


def test_keywords_with_surrounding_whitespace(raw: dict[str, str]) -> None:
    padded = {f"  {k}  ": v for k, v in raw.items()}

    s = ConvertSettings.model_validate(padded)

    assert s.iss == "transient"


def test_keywords_differing_only_by_case_are_a_duplicate(
    tmp_path: Path, raw: dict[str, str]
) -> None:
    ini = tmp_path / "dup.ini"
    ini.write_text(
        "\n".join(f"{k}={v}" for k, v in raw.items()) + "\niss=steady-state\n"
    )

    with pytest.raises(configparser.DuplicateOptionError):
        read_convert_settings(ini)


# defaults and value normalisation
def test_defaults(raw: dict[str, str]) -> None:
    s = ConvertSettings.model_validate(raw)

    assert s.model_name == "imported"
    assert s.interval == "D"
    assert s.start_date == date(1970, 1, 1)
    assert s.end_date == date(1970, 1, 2)
    assert s.overwrite is False
    assert s.bbox is None
    assert s.cellsize is None
    assert s.bin_dir is None
    assert s.msw_dbase is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("transient", "transient"),
        ("TRANSIENT", "transient"),
        ("  Transient  ", "transient"),
        ("steady-state", "steady-state"),
        ("Steady-State", "steady-state"),
    ],
)
def test_iss_is_normalised(raw: dict[str, str], value: str, expected: str) -> None:
    raw["ISS"] = value

    s = ConvertSettings.model_validate(raw)

    assert s.iss == expected
    assert s.is_steady_state == (expected == "steady-state")


@pytest.mark.parametrize("value", ["1", "steady_state", "steadystate", "static"])
def test_iss_invalid_value_raises(raw: dict[str, str], value: str) -> None:
    raw["ISS"] = value

    with pytest.raises(ValidationError):
        ConvertSettings.model_validate(raw)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ],
)
def test_overwrite_is_parsed_as_bool(
    raw: dict[str, str], value: str, expected: bool
) -> None:
    raw["OVERWRITE"] = value

    assert ConvertSettings.model_validate(raw).overwrite is expected


def test_overwrite_invalid_raises(raw: dict[str, str]) -> None:
    raw["OVERWRITE"] = "maybe"

    with pytest.raises(ValidationError):
        ConvertSettings.model_validate(raw)


def test_dates_are_parsed_as_dates(raw: dict[str, str]) -> None:
    raw.update({"SDATE": "2011-01-01", "EDATE": "2011-02-03"})

    s = ConvertSettings.model_validate(raw)

    assert s.start_date == date(2011, 1, 1)
    assert s.end_date == date(2011, 2, 3)
    assert type(s.start_date) is date


def test_invalid_date_raises(raw: dict[str, str]) -> None:
    raw["SDATE"] = "01-01-2011x"

    with pytest.raises(ValidationError):
        ConvertSettings.model_validate(raw)


def test_percent_in_value_is_not_interpolated(
    tmp_path: Path, raw: dict[str, str]
) -> None:
    raw["MODELNAME"] = "peel%venen"
    ini = write_ini(tmp_path / "pct.ini", raw)

    assert read_convert_settings(ini).model_name == "peel%venen"


# blank values
@pytest.mark.parametrize(
    "key",
    [
        "COUPLER_DIR",
        "MSW_DBASE",
        "WINDOW",
        "CELLSIZE",
        "SDATE",
        "EDATE",
        "INTERVAL",
        "MODELNAME",
        "OVERWRITE",
    ],
)
def test_blank_optional_value_counts_as_not_provided(
    tmp_path: Path, raw: dict[str, str], key: str
) -> None:
    raw[key] = ""
    ini = write_ini(tmp_path / "blank.ini", raw)

    s = read_convert_settings(ini)

    defaults = ConvertSettings.model_validate({k: v for k, v in raw.items() if k != key})
    assert s == defaults


def test_whitespace_only_value_counts_as_blank(raw: dict[str, str]) -> None:
    raw["COUPLER_DIR"] = "   "

    assert ConvertSettings.model_validate(raw).bin_dir is None


@pytest.mark.parametrize("key", ["PRJFILE_IN", "OUTPUT_FOLDER", "ISS"])
def test_blank_required_value_raises(raw: dict[str, str], key: str) -> None:
    raw[key] = ""

    with pytest.raises(ValidationError) as exc:
        ConvertSettings.model_validate(raw)

    assert exc.value.errors()[0]["type"] == "missing"


# window and cellsize
def test_window_is_parsed_to_floats(raw: dict[str, str]) -> None:
    raw.update({"WINDOW": "175000, 364000 ,201000,397000.5", "CELLSIZE": "100"})

    s = ConvertSettings.model_validate(raw)

    assert s.bbox == (175000.0, 364000.0, 201000.0, 397000.5)
    assert all(isinstance(x, float) for x in s.bbox)


@pytest.mark.parametrize("window", ["1,2,3", "1,2,3,4,5", "a,b,c,d", "1,2,3,"])
def test_window_malformed_raises(raw: dict[str, str], window: str) -> None:
    raw.update({"WINDOW": window, "CELLSIZE": "100"})

    with pytest.raises(ValidationError):
        ConvertSettings.model_validate(raw)


@pytest.mark.parametrize("window", ["10,0,5,10", "0,10,10,5", "5,5,5,10", "0,5,10,5"])
def test_window_with_inverted_or_empty_extent_raises(
    raw: dict[str, str], window: str
) -> None:
    raw.update({"WINDOW": window, "CELLSIZE": "100"})

    with pytest.raises(ValidationError, match="xmin < xmax and ymin < ymax"):
        ConvertSettings.model_validate(raw)


def test_window_without_cellsize_raises(raw: dict[str, str]) -> None:
    raw["WINDOW"] = "0,0,10,10"

    with pytest.raises(ValidationError, match="provided together"):
        ConvertSettings.model_validate(raw)


def test_cellsize_without_window_raises(raw: dict[str, str]) -> None:
    raw["CELLSIZE"] = "100"

    with pytest.raises(ValidationError, match="provided together"):
        ConvertSettings.model_validate(raw)


@pytest.mark.parametrize("cellsize", ["0", "-5", "abc"])
def test_invalid_cellsize_raises(raw: dict[str, str], cellsize: str) -> None:
    raw.update({"WINDOW": "0,0,10,10", "CELLSIZE": cellsize})

    with pytest.raises(ValidationError):
        ConvertSettings.model_validate(raw)


def test_float_cellsize_is_accepted(raw: dict[str, str]) -> None:
    raw.update({"WINDOW": "0,0,10,10", "CELLSIZE": "12.5"})

    assert ConvertSettings.model_validate(raw).cellsize == 12.5


# date ordering
def test_end_before_start_raises_for_transient(raw: dict[str, str]) -> None:
    raw.update({"SDATE": "2011-01-07", "EDATE": "2011-01-01"})

    with pytest.raises(ValidationError, match="EDATE must not be before SDATE"):
        ConvertSettings.model_validate(raw)


def test_end_equal_to_start_is_allowed(raw: dict[str, str]) -> None:
    raw.update({"SDATE": "2011-01-01", "EDATE": "2011-01-01"})

    ConvertSettings.model_validate(raw)


def test_date_order_is_not_checked_for_steady_state(raw: dict[str, str]) -> None:
    raw.update({"ISS": "steady-state", "SDATE": "2011-01-07", "EDATE": "2011-01-01"})

    assert ConvertSettings.model_validate(raw).is_steady_state


# paths
def test_missing_prjfile_raises(tmp_path: Path, raw: dict[str, str]) -> None:
    raw["PRJFILE_IN"] = str(tmp_path / "nope.PRJ")

    with pytest.raises(ValidationError) as exc:
        ConvertSettings.model_validate(raw)

    assert exc.value.errors()[0]["loc"] == ("prjfile_in",)


def test_prjfile_pointing_to_directory_raises(
    tmp_path: Path, raw: dict[str, str]
) -> None:
    raw["PRJFILE_IN"] = str(tmp_path)

    with pytest.raises(ValidationError):
        ConvertSettings.model_validate(raw)


def test_nonexistent_coupler_dir_raises(tmp_path: Path, raw: dict[str, str]) -> None:
    raw["COUPLER_DIR"] = str(tmp_path / "does_not_exist")

    with pytest.raises(ValidationError):
        ConvertSettings.model_validate(raw)


def test_nonexistent_msw_dbase_is_accepted_at_load(
    tmp_path: Path, raw: dict[str, str]
) -> None:
    # Only needed, and only checked, for projects with a CAP entry (validate_msw).
    raw["MSW_DBASE"] = str(tmp_path / "does_not_exist")

    ConvertSettings.model_validate(raw)


def test_nonempty_output_folder_raises_without_overwrite(raw: dict[str, str]) -> None:
    out = Path(raw["OUTPUT_FOLDER"])
    out.mkdir()
    (out / "old.txt").touch()

    with pytest.raises(ValidationError, match="Output folder is not empty"):
        ConvertSettings.model_validate(raw)


def test_nonempty_output_folder_is_allowed_with_overwrite(raw: dict[str, str]) -> None:
    out = Path(raw["OUTPUT_FOLDER"])
    out.mkdir()
    (out / "old.txt").touch()
    raw["OVERWRITE"] = "true"

    assert ConvertSettings.model_validate(raw).overwrite is True


def test_empty_existing_output_folder_is_allowed(raw: dict[str, str]) -> None:
    Path(raw["OUTPUT_FOLDER"]).mkdir()

    ConvertSettings.model_validate(raw)


def test_relative_output_folder_is_checked_at_ini_location(
    tmp_path: Path, ini_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_dir_tree(ini_dir, files=["model.PRJ", "../out/old.txt"], dirs=[])
    ini = write_ini(
        ini_dir / "c.ini",
        {"PRJFILE_IN": "model.PRJ", "OUTPUT_FOLDER": "../out", "ISS": "transient"},
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with pytest.raises(ValidationError, match="Output folder is not empty"):
        read_convert_settings(ini)


def test_output_folder_does_not_need_to_exist(
    tmp_path: Path, raw: dict[str, str]
) -> None:
    raw["OUTPUT_FOLDER"] = str(tmp_path / "new" / "nested")

    assert ConvertSettings.model_validate(raw).out_dir == tmp_path / "new" / "nested"


# MetaSWAP checks 
def test_validate_msw_passes_with_both_paths(
    tmp_path: Path, raw: dict[str, str]
) -> None:
    raw.update(
        {"COUPLER_DIR": str(tmp_path / "bin"), "MSW_DBASE": str(tmp_path / "msw")}
    )

    ConvertSettings.model_validate(raw).validate_msw()


def test_validate_msw_requires_database(tmp_path: Path, raw: dict[str, str]) -> None:
    raw["COUPLER_DIR"] = str(tmp_path / "bin")

    with pytest.raises(ValueError, match="MetaSwap database"):
        ConvertSettings.model_validate(raw).validate_msw()


def test_validate_msw_requires_coupler_dir(tmp_path: Path, raw: dict[str, str]) -> None:
    raw["MSW_DBASE"] = str(tmp_path / "msw")

    with pytest.raises(ValueError, match="Coupler binaries"):
        ConvertSettings.model_validate(raw).validate_msw()


def test_validate_msw_requires_existing_database(
    tmp_path: Path, raw: dict[str, str]
) -> None:
    raw.update(
        {
            "COUPLER_DIR": str(tmp_path / "bin"),
            "MSW_DBASE": str(tmp_path / "does_not_exist"),
        }
    )

    with pytest.raises(FileNotFoundError, match="MetaSwap database"):
        ConvertSettings.model_validate(raw).validate_msw()


def test_msw_paths_are_not_required_to_load_settings(raw: dict[str, str]) -> None:
    # Only validate_msw() enforces them, for projects with a CAP entry.
    ConvertSettings.model_validate(raw)


# strictness and file handling
def test_unknown_keyword_raises(raw: dict[str, str]) -> None:
    raw["COUPLER_DIRR"] = "typo"

    with pytest.raises(ValidationError) as exc:
        ConvertSettings.model_validate(raw)

    assert exc.value.errors()[0]["type"] == "extra_forbidden"


def test_read_settings_accepts_str_path(tmp_path: Path, raw: dict[str, str]) -> None:
    ini = write_ini(tmp_path / "c.ini", raw)

    assert read_convert_settings(str(ini)).iss == "transient"


def test_missing_ini_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_convert_settings(tmp_path / "missing.ini")


def test_empty_ini_file_raises(tmp_path: Path) -> None:
    ini = tmp_path / "empty.ini"
    ini.write_text("")

    with pytest.raises(KeyError):
        read_convert_settings(ini)


def test_ini_with_crlf_line_endings(tmp_path: Path, raw: dict[str, str]) -> None:
    ini = tmp_path / "crlf.ini"
    ini.write_bytes("\r\n".join(f"{k}={v}" for k, v in raw.items()).encode())

    assert read_convert_settings(ini).iss == "transient"

def test_simulation_times(raw: dict[str, str]) -> None:
    raw.update(
        {
            "SDATE": "2024-01-01",
            "EDATE": "2024-01-03",
            "INTERVAL": "1D",
        }
    )

    settings = ConvertSettings.model_validate(raw)
    times = settings.simulation_times()
    assert times == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]

@pytest.mark.parametrize(
    "log_level",
    [
        "invalid",
        22,
        22.0,
        "33",
    ],
)
def test_invalid_log_level_raises(raw: dict[str, str], log_level: str | int) -> None:
    raw["LOGLEVEL"] = log_level

    with pytest.raises(ValueError, match="LOGLEVEL must be one of"):
        ConvertSettings.model_validate(raw)

@pytest.mark.parametrize(
    "log_level",
    [
        "debug",
        "info",
        "warning",
        "error",
        "critical",
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
        "10",
        "20",   
        "30",
        "40",
        "50",
        10,
        20,
        30,
        40,
        50,
    ],
)
def test_valid_log_levels(raw: dict[str, str], log_level: str) -> None:
    raw["LOGLEVEL"] = log_level
    settings = ConvertSettings.model_validate(raw)
    assert isinstance(settings.log_level, int)