"""Tests for the GENPUZZLE settings, read from an ini file or validated directly."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from settings import GenPuzzleSettings, read_genpuzzle_settings


@pytest.fixture
def genfile_in(tmp_path: Path) -> Path:
    path = tmp_path / "in.gen"
    path.touch()
    return path


@pytest.fixture
def raw(genfile_in: Path, tmp_path: Path) -> dict[str, str]:
    return {
        "FUNCTION": "GENPUZZLE",
        "GENFILE_IN": str(genfile_in),
        "GENFILE_OUT": str(tmp_path / "out.gen"),
    }


def write_ini(path: Path, entries: dict[str, str]) -> Path:
    lines = [f"{key}={value}" for key, value in entries.items()]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_defaults(raw: dict[str, str]) -> None:
    settings = GenPuzzleSettings.model_validate(raw)

    assert settings.function == "GENPUZZLE"
    assert settings.tolerance == 0.001
    assert settings.overwrite is False


@pytest.mark.parametrize("value", ["GENPUZZLE", "genpuzzle", "  GenPuzzle  "])
def test_function_is_normalised(raw: dict[str, str], value: str) -> None:
    raw["FUNCTION"] = value

    assert GenPuzzleSettings.model_validate(raw).function == "GENPUZZLE"


def test_different_function_raises(raw: dict[str, str]) -> None:
    raw["FUNCTION"] = "CONVERT"

    with pytest.raises(ValidationError) as exc:
        GenPuzzleSettings.model_validate(raw)

    assert exc.value.errors()[0]["loc"] == ("function",)


def test_different_tolerance_is_parsed(raw: dict[str, str]) -> None:
    raw["TOLERANCE"] = "0.0001"

    assert GenPuzzleSettings.model_validate(raw).tolerance == 0.0001


def test_existing_output_raises_without_overwrite(raw: dict[str, str]) -> None:
    Path(raw["GENFILE_OUT"]).touch()

    with pytest.raises(ValidationError, match="Output GEN file already exists"):
        GenPuzzleSettings.model_validate(raw)


def test_existing_output_is_allowed_with_overwrite(raw: dict[str, str]) -> None:
    Path(raw["GENFILE_OUT"]).touch()
    raw["OVERWRITE"] = "true"

    assert GenPuzzleSettings.model_validate(raw).overwrite is True


def test_missing_input_raises(raw: dict[str, str], tmp_path: Path) -> None:
    raw["GENFILE_IN"] = str(tmp_path / "missing.gen")

    with pytest.raises(ValidationError) as exc:
        GenPuzzleSettings.model_validate(raw)

    assert exc.value.errors()[0]["loc"] == ("genfile_in",)


def test_read_genpuzzle_settings_resolves_relative_paths_against_ini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ini_dir = tmp_path / "inifiles"
    ini_dir.mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "in.gen").touch()
    ini = write_ini(
        ini_dir / "genpuzzle.ini",
        {
            "FUNCTION": "GENPUZZLE",
            "GENFILE_IN": "../data/in.gen",
            "GENFILE_OUT": "../data/out.gen",
            "TOLERANCE": "0.01",
        },
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    settings = read_genpuzzle_settings(ini)

    root = tmp_path.resolve()
    assert settings.genfile_in == root / "data" / "in.gen"
    assert settings.genfile_out == root / "data" / "out.gen"
    assert settings.tolerance == 0.01


def test_read_genpuzzle_settings_from_ini_with_absolute_paths(
    tmp_path: Path, genfile_in: Path
) -> None:
    ini = write_ini(
        tmp_path / "genpuzzle_test.ini",
        {
            "FUNCTION": "GENPUZZLE",
            "GENFILE_IN": str(genfile_in),
            "GENFILE_OUT": str(tmp_path / "out.gen"),
        },
    )

    settings = read_genpuzzle_settings(str(ini))

    assert settings.genfile_in == genfile_in.resolve()
    assert settings.genfile_out == (tmp_path / "out.gen").resolve()
