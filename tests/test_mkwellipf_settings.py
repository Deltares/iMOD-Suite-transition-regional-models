"""Tests for the MKWELLIPF settings, read from an ini file."""

import logging
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.settings import MkWellIpfSettings, read_mkwellipf_settings

NLAY = 2


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A folder with the files an ini file can refer to."""
    names = ["wells.ipf", "wells2.ipf"]
    names += [f"{kind}_l{i}.idf" for kind in ("top", "bot", "khv", "kd") for i in (1, 2)]
    for name in names:
        (tmp_path / name).touch()
    return tmp_path


@pytest.fixture
def raw() -> dict[str, str]:
    return {
        "FUNCTION": "MKWELLIPF",
        "NIPF": "1",
        "IPF1": "wells.ipf",
        "NLAY": str(NLAY),
        "TOPIDF1": "top_l1.idf",
        "TOPIDF2": "top_l2.idf",
        "BOTIDF1": "bot_l1.idf",
        "BOTIDF2": "bot_l2.idf",
    }


def write_ini(path: Path, entries: dict[str, str]) -> Path:
    path.write_text("\n".join(f"{key}={value}" for key, value in entries.items()), encoding="utf-8")
    return path


def read(folder: Path, entries: dict[str, str]) -> MkWellIpfSettings:
    return read_mkwellipf_settings(write_ini(folder / "mkwellipf.ini", entries))


def test_defaults(folder: Path, raw: dict[str, str]) -> None:
    settings = read(folder, raw)

    assert settings.ipf_files == [folder / "wells.ipf"]
    assert settings.top_files == [folder / "top_l1.idf", folder / "top_l2.idf"]
    assert settings.bottom_files == [folder / "bot_l1.idf", folder / "bot_l2.idf"]
    assert settings.khv_files is None
    assert settings.kd_files is None
    assert settings.iss == 0
    assert (settings.x_col, settings.y_col, settings.q_col) == (1, 2, 1)
    assert (settings.top_col, settings.bottom_col) == (4, 5)
    assert (settings.fnodata, settings.hnodata) == (-99999.0, 0.0)
    assert settings.log_level == logging.INFO
    assert not settings.overwrite


def test_keywords_are_not_case_sensitive_and_function_is_normalised(folder: Path, raw: dict[str, str]) -> None:
    entries = {key.lower(): value for key, value in raw.items()}
    entries["function"] = "mkwellipf"

    assert read(folder, entries).function == "MKWELLIPF"


def test_different_function_raises(folder: Path, raw: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="FUNCTION must be MKWELLIPF"):
        read(folder, {**raw, "FUNCTION": "GENPUZZLE"})


def test_without_layers(folder: Path) -> None:
    settings = read(folder, {"NIPF": "2", "IPF1": "wells.ipf", "IPF2": "wells2.ipf"})

    assert settings.ipf_files == [folder / "wells.ipf", folder / "wells2.ipf"]
    assert settings.top_files is None


def test_k_files_and_constants(folder: Path, raw: dict[str, str]) -> None:
    settings = read(folder, {**raw, "KHVIDF1": "khv_l1.idf", "KHVIDF2": "2.5"})

    assert settings.khv_files == [folder / "khv_l1.idf", 2.5]


def test_transmissivity_and_minkd(folder: Path, raw: dict[str, str]) -> None:
    settings = read(folder, {**raw, "KDVIDF1": "kd_l1.idf", "KDVIDF2": "20", "MINKD": "40"})

    assert settings.kd_files == [folder / "kd_l1.idf", 20.0]
    assert settings.khv_files is None
    assert settings.minkd == 40.0


def test_other_keywords_are_parsed(folder: Path, raw: dict[str, str]) -> None:
    entries = {
        **raw, "ISS": "1", "SDATE": "20200101", "EDATE": "2020-12-31", "IXCOL": "3", "IYCOL": "4",
        "IQCOL": "6", "ITCOL": "7", "IBCOL": "8", "MINKHT": "1.5", "FNODATA": "-999",
        "HNODATA": "-1", "OVERWRITE": "true", "LOGLEVEL": "warning",
    }

    settings = read(folder, entries)

    assert settings.iss == 1
    assert (settings.sdate, settings.edate) == (date(2020, 1, 1), date(2020, 12, 31))
    assert (settings.x_col, settings.y_col, settings.q_col) == (3, 4, 6)
    assert (settings.top_col, settings.bottom_col) == (7, 8)
    assert settings.minkht == 1.5
    assert (settings.fnodata, settings.hnodata) == (-999.0, -1.0)
    assert settings.overwrite
    assert settings.log_level == logging.WARNING


def test_relative_paths_are_resolved_against_the_ini_file(folder: Path, raw: dict[str, str]) -> None:
    (folder / "sub").mkdir()
    entries = {key: f"../{value}" if value.endswith((".ipf", ".idf")) else value for key, value in raw.items()}
    ini = write_ini(folder / "sub" / "mkwellipf.ini", entries)

    settings = read_mkwellipf_settings(ini)

    assert settings.ipf_files[0].samefile(folder / "wells.ipf")
    assert settings.top_files[0].samefile(folder / "top_l1.idf")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"IPF1": "missing.ipf"}, "does not point to a file"),
        ({"NIPF": "2"}, "IPF2 is missing"),
        ({"NIPF": None}, "NIPF and IPF1 to IPFN are required"),
        ({"TOPIDF2": None}, "TOPIDF2 is missing"),
        ({"BOTIDF1": None}, "BOTIDF1 is missing"),
        ({"ISS": "1"}, "SDATE and EDATE are required"),
        ({"ISS": "2"}, "Input should be 0 or 1"),
        ({"IMIDF": "1"}, "IMIDF is not supported"),
        ({"KHVIDF1": "khv_l1.idf"}, "every layer"),
        ({"KHVIDF1": "khv_l1.idf", "KDVIDF2": "5"}, "not both"),
        ({"SDATE": "20200201", "EDATE": "20200101"}, "EDATE must not be before SDATE"),
        ({"LOGLEVEL": "loud"}, "LOGLEVEL"),
        ({"UNKNOWN": "1"}, "Extra inputs"),
    ],
)
def test_invalid_input_raises(folder: Path, raw: dict[str, str], change: dict, message: str) -> None:
    entries = {**raw, **change}
    entries = {key: value for key, value in entries.items() if value is not None}
    if "NIPF" in change and change["NIPF"] is None:
        entries.pop("IPF1")

    with pytest.raises(ValidationError, match=message):
        read(folder, entries)


def test_output_folder_is_resolved_against_the_ini_file(folder: Path, raw: dict[str, str]) -> None:
    settings = read(folder, {**raw, "OUTPUT_FOLDER": "results"})

    assert settings.output_folder == folder / "results"
    assert read(folder, raw).output_folder is None


def test_ipf_files_need_different_names_with_an_output_folder(folder: Path) -> None:
    (folder / "sub").mkdir()
    (folder / "sub" / "wells.ipf").touch()
    entries = {"NIPF": "2", "IPF1": "wells.ipf", "IPF2": "sub/wells.ipf"}

    assert len(read(folder, entries).ipf_files) == 2
    with pytest.raises(ValidationError, match="different names"):
        read(folder, {**entries, "OUTPUT_FOLDER": "results"})


def test_python_call_with_lists_still_works(folder: Path) -> None:
    settings = MkWellIpfSettings(ipf_files=[folder / "wells.ipf"], iss=0)

    assert settings.top_files is None
    assert settings.function == "MKWELLIPF"


def test_top_and_bottom_need_the_same_number_of_layers(folder: Path) -> None:
    with pytest.raises(ValidationError, match="same number of layers"):
        MkWellIpfSettings(ipf_files=[folder / "wells.ipf"], top_files=[folder / "top_l1.idf"])


def test_kd_files_must_cover_every_layer_given_as_lists(folder: Path) -> None:
    with pytest.raises(ValidationError, match="every layer"):
        MkWellIpfSettings(
            ipf_files=[folder / "wells.ipf"],
            top_files=[folder / "top_l1.idf"],
            bottom_files=[folder / "bot_l1.idf"],
            kd_files=[1.0, 2.0],
        )
