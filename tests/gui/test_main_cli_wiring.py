from pathlib import Path

import pytest

from rheojax.gui.main import parse_args


def test_protocol_required_with_import():
    with pytest.raises(SystemExit):
        parse_args(["--import", "data.csv"])  # no --protocol


def test_protocol_without_import_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--protocol", "oscillation"])  # no --import


def test_project_and_import_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--project", "p.rheojax", "--import", "data.csv", "--protocol", "oscillation"])


def test_import_with_protocol_parses_successfully():
    args = parse_args(["--import", "data.csv", "--protocol", "oscillation"])
    assert args.import_file == Path("data.csv")
    assert args.protocol == "oscillation"


def test_project_alone_still_parses():
    args = parse_args(["--project", "p.rheojax"])
    assert args.project == Path("p.rheojax")
    assert args.import_file is None
    assert args.protocol is None
