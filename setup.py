#!/usr/bin/env python
"""Setup script for rheojax package - backward compatibility support."""

from setuptools import setup

# The actual package configuration is in pyproject.toml
# This file exists only for backward compatibility with older pip versions
# and tools that still expect setup.py

if __name__ == "__main__":
    setup()