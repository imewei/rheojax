import pytest

from rheojax.gui.foundation.import_service import import_dataset


def test_import_dataset_sets_protocol_and_metadata(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "omega,G',G''\n0.1,100.0,10.0\n1.0,150.0,20.0\n10.0,200.0,50.0\n"
    )

    ref, data = import_dataset(csv_path, "oscillation")

    assert ref.protocol_type == "oscillation"
    assert ref.origin == "imported"
    assert ref.id  # non-empty uuid4 hex
    assert data.test_mode == "oscillation"


def test_import_dataset_rejects_unknown_protocol(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n0.1,10\n")
    with pytest.raises(ValueError):
        import_dataset(csv_path, "not_a_real_protocol")


def test_import_dataset_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_dataset(tmp_path / "does_not_exist.csv", "oscillation")


def test_import_dataset_rejects_wrong_column_count(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("time,stress\n0.1,10\n1.0,20\n10.0,30\n")
    with pytest.raises(ValueError):
        import_dataset(csv_path, "oscillation")
