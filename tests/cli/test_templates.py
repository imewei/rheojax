"""Tests for rheojax.cli._templates — YAML pipeline template library."""

from __future__ import annotations

import pytest
import yaml

from rheojax.cli._templates import get_template, list_templates, write_template


class TestListTemplates:
    @pytest.mark.smoke
    def test_list_templates_returns_non_empty(self):
        templates = list_templates()
        assert len(templates) > 0

    @pytest.mark.unit
    def test_list_templates_entries_have_name_and_description(self):
        for entry in list_templates():
            assert "name" in entry
            assert "description" in entry
            assert isinstance(entry["name"], str)
            assert isinstance(entry["description"], str)


class TestGetTemplate:
    @pytest.mark.smoke
    def test_get_template_returns_valid_yaml(self):
        for entry in list_templates():
            yaml_str = get_template(entry["name"])
            parsed = yaml.safe_load(yaml_str)
            assert isinstance(parsed, dict)

    @pytest.mark.unit
    def test_get_template_basic_exists(self):
        yaml_str = get_template("basic")
        assert yaml_str  # Non-empty string

    @pytest.mark.unit
    def test_get_template_bayesian_exists(self):
        yaml_str = get_template("bayesian")
        parsed = yaml.safe_load(yaml_str)
        step_types = [s["type"] for s in parsed.get("steps", [])]
        assert "bayesian" in step_types

    @pytest.mark.unit
    def test_get_template_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_template("nonexistent_template_xyz")

    @pytest.mark.unit
    def test_each_template_has_version_and_steps(self):
        for entry in list_templates():
            parsed = yaml.safe_load(get_template(entry["name"]))
            assert "version" in parsed
            assert "steps" in parsed


class TestWriteTemplate:
    @pytest.mark.smoke
    def test_write_template_creates_file(self, tmp_path):
        output = tmp_path / "my_pipeline.yaml"
        write_template("basic", output)
        assert output.exists()
        assert output.stat().st_size > 0

    @pytest.mark.unit
    def test_write_template_content_is_valid_yaml(self, tmp_path):
        output = tmp_path / "out.yaml"
        write_template("basic", output)
        parsed = yaml.safe_load(output.read_text())
        assert isinstance(parsed, dict)

    @pytest.mark.unit
    def test_write_template_unknown_raises(self, tmp_path):
        with pytest.raises(KeyError):
            write_template("does_not_exist", tmp_path / "out.yaml")
