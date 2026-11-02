"""Tests for the GENSNAPTOGRID settings, read from an ini file or validated directly."""

import logging
from pathlib import Path

import imod
import numpy as np
import pytest
from imod.util import empty_2d
from pydantic import ValidationError

from src.settings import GenSnapToGridSettings, read_gen_snap_to_grid_settings

imod_logger = logging.getLogger("imod")


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def raw(tmp_path: Path) -> dict[str, str]:
    genfile_in = tmp_path / "in.gen"
    genfile_in.touch()
    return {
        "FUNCTION": "GENSNAPTOGRID",
        "GENFILE_IN": str(genfile_in),
        "GENFILE_OUT": str(tmp_path / "out.gen"),
        "WINDOW": "0,0,4,4",
        "CELLSIZE": "1",
    }


@pytest.fixture
def idf_files(tmp_path: Path) -> tuple[Path, Path]:
    top, bot = tmp_path / "top.idf", tmp_path / "bot.idf"
    top.touch()
    bot.touch()
    return top, bot


def write_ini(path: Path, entries: dict[str, str]) -> Path:
    lines = [f"{key}={value}" for key, value in entries.items()]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_defaults(raw: dict[str, str]) -> None:
    settings = GenSnapToGridSettings.model_validate(raw)

    assert settings.i3d == 0
    assert settings.idf_top is None
    assert settings.idf_bot is None
    assert settings.max_snap_distance == 0.0


@pytest.mark.parametrize("value", ["0", "0.5", "-1.0"])
def test_max_snap_distance_must_not_be_negative(raw: dict[str, str], value: str) -> None:
    raw["MAX_SNAP_DISTANCE"] = value

    if float(value) < 0.0:
        with pytest.raises(ValidationError):
            GenSnapToGridSettings.model_validate(raw)
    else:
        settings = GenSnapToGridSettings.model_validate(raw)
        assert settings.max_snap_distance == float(value)


def test_i3d_one_accepts_top_and_bottom(
    raw: dict[str, str], idf_files: tuple[Path, Path]
) -> None:
    top, bot = idf_files
    raw.update({"I3D": "1", "IDF_TOP": str(top), "IDF_BOT": str(bot)})

    settings = GenSnapToGridSettings.model_validate(raw)

    assert settings.i3d == 1
    assert settings.idf_top == top
    assert settings.idf_bot == bot


@pytest.mark.parametrize("missing", [["IDF_TOP"], ["IDF_BOT"], ["IDF_TOP", "IDF_BOT"]])
def test_i3d_requires_top_and_bottom(
    raw: dict[str, str], idf_files: tuple[Path, Path], missing: list[str]
) -> None:
    top, bot = idf_files
    raw.update({"I3D": "1", "IDF_TOP": str(top), "IDF_BOT": str(bot)})
    for key in missing:
        del raw[key]

    with pytest.raises(ValidationError, match="IDF_TOP and IDF_BOT"):
        GenSnapToGridSettings.model_validate(raw)


@pytest.mark.parametrize("key", ["IDF_TOP", "IDF_BOT"])
def test_i3d_zero_rejects_top_and_bottom(
    raw: dict[str, str], idf_files: tuple[Path, Path], key: str
) -> None:
    raw[key] = str(idf_files[0])

    with pytest.raises(ValidationError, match=key):
        GenSnapToGridSettings.model_validate(raw)


@pytest.mark.parametrize("value", ["2", "-1", "yes"])
def test_i3d_other_values_are_rejected(raw: dict[str, str], value: str) -> None:
    raw["I3D"] = value

    with pytest.raises(ValidationError, match="I3D"):
        GenSnapToGridSettings.model_validate(raw)


def test_idf_paths_resolve_against_ini_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ini_dir = tmp_path / "inifiles"
    ini_dir.mkdir()
    (tmp_path / "data").mkdir()
    for name in ("in.gen", "grid.idf", "top.idf", "bot.idf"):
        (tmp_path / "data" / name).touch()
    ini = write_ini(
        ini_dir / "snap.ini",
        {
            "FUNCTION": "GENSNAPTOGRID",
            "GENFILE_IN": "../data/in.gen",
            "GENFILE_OUT": "../data/out.gen",
            "IDFFILE": "../data/grid.idf",
            "I3D": "1",
            "IDF_TOP": "../data/top.idf",
            "IDF_BOT": "../data/bot.idf",
        },
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    settings = read_gen_snap_to_grid_settings(ini)

    root = tmp_path.resolve() / "data"
    assert settings.idffile == root / "grid.idf"
    assert settings.idf_top == root / "top.idf"
    assert settings.idf_bot == root / "bot.idf"


def test_cell_size_is_accepted_as_alias(raw: dict[str, str]) -> None:
    del raw["CELLSIZE"]
    raw["CELL_SIZE"] = "2"

    settings = GenSnapToGridSettings.model_validate(raw)

    assert settings.cellsize == 2.0


def test_window_without_cellsize_is_rejected(raw: dict[str, str]) -> None:
    del raw["CELLSIZE"]

    with pytest.raises(ValidationError, match="WINDOW"):
        GenSnapToGridSettings.model_validate(raw)


def test_window_with_idffile_ignores_the_idf_for_the_grid(
    raw: dict[str, str], idf_files: tuple[Path, Path]
) -> None:
    raw["IDFFILE"] = str(idf_files[0])
    handler = _ListHandler()
    imod_logger.addHandler(handler)
    try:
        settings = GenSnapToGridSettings.model_validate(raw)
    finally:
        imod_logger.removeHandler(handler)

    assert settings.like is None
    assert settings.window == (0.0, 0.0, 4.0, 4.0)
    assert settings.cellsize == 1.0
    assert any("IDFFILE" in message for message in handler.messages)


def test_idffile_with_cellsize_but_no_window_is_rejected(
    raw: dict[str, str], idf_files: tuple[Path, Path]
) -> None:
    del raw["WINDOW"]
    raw["IDFFILE"] = str(idf_files[0])

    with pytest.raises(ValidationError, match="CELLSIZE"):
        GenSnapToGridSettings.model_validate(raw)


def test_output_format_is_not_a_keyword(raw: dict[str, str]) -> None:
    raw["OUTPUT_FORMAT"] = "imod5"

    with pytest.raises(ValidationError, match="(?i)output_format"):
        GenSnapToGridSettings.model_validate(raw)


def test_window_and_like_follow_the_settings(tmp_path: Path, raw: dict[str, str]) -> None:
    settings = GenSnapToGridSettings.model_validate(raw)
    assert settings.window == (0.0, 0.0, 4.0, 4.0)
    assert settings.like is None

    grid = empty_2d(dx=1.0, xmin=0.0, xmax=4.0, dy=-1.0, ymin=0.0, ymax=4.0)
    imod.idf.save(tmp_path / "grid", grid.astype(np.float32))
    idf = next(tmp_path.glob("grid*.idf"))
    del raw["WINDOW"], raw["CELLSIZE"]
    raw["IDFFILE"] = str(idf)

    from_idf = GenSnapToGridSettings.model_validate(raw)

    assert from_idf.window is None
    assert from_idf.like.dims == ("y", "x")
    np.testing.assert_allclose(from_idf.like["x"], grid["x"])
    np.testing.assert_allclose(from_idf.like["y"], grid["y"])
