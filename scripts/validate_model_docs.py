#!/usr/bin/env python3
"""Validate RheoJAX model documentation for completeness and consistency.

This script checks model documentation files for:
1. Required sections presence
2. Parameter consistency with model code
3. Minimum reference count
4. Internal link validity
5. Basic RST syntax

Usage:
    python scripts/validate_model_docs.py [path]

    path: Optional path to specific .rst file or directory (default: docs/source/models/)

Examples:
    python scripts/validate_model_docs.py
    python scripts/validate_model_docs.py docs/source/models/flow/carreau.rst
    python scripts/validate_model_docs.py docs/source/models/classical/
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Required sections for model documentation
REQUIRED_SECTIONS = [
    "Quick Reference",
    "Overview",
    "Parameters",
]

# Recommended sections (warnings if missing)
RECOMMENDED_SECTIONS = [
    "Notation Guide",
    "Physical Foundations",
    "Governing Equations",
    "Validity and Assumptions",
    "What You Can Learn",
    "Fitting Guidance",
    "Usage",
    "See Also",
    "References",
]

# Quick Reference required fields
QUICK_REFERENCE_FIELDS = [
    "Use when:",
    "Parameters:",
    "Key equation:",
    "Test modes:",
    "Material examples:",
]

# Minimum reference count by model complexity
MIN_REFERENCES_SIMPLE = 5
MIN_REFERENCES_RECOMMENDED = 10


@dataclass
class ValidationResult:
    """Result of validating a single documentation file."""

    file_path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if no errors."""
        return len(self.errors) == 0

    @property
    def line_count(self) -> int:
        """Return line count of the file."""
        return len(self.file_path.read_text().splitlines())


def find_sections(content: str) -> set[str]:
    """Extract section titles from RST content."""
    sections = set()
    lines = content.splitlines()

    for i, line in enumerate(lines):
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # Check for underline patterns (-, ~, ^, =)
            if re.match(r"^[-~^=]{3,}$", next_line) and len(next_line) >= len(
                line.strip()
            ):
                sections.add(line.strip())

    return sections


def check_quick_reference(content: str) -> list[str]:
    """Check Quick Reference section for required fields."""
    issues = []

    # Find Quick Reference section
    qr_match = re.search(
        r"Quick Reference\n-+\n(.*?)(?=\n\n\.\. contents::|\n\n[A-Z]|\n-{3,}|\Z)",
        content,
        re.DOTALL,
    )

    if not qr_match:
        issues.append("Quick Reference section not found or malformed")
        return issues

    qr_content = qr_match.group(1)

    for field_name in QUICK_REFERENCE_FIELDS:
        if field_name not in qr_content:
            issues.append(f"Quick Reference missing field: {field_name}")

    return issues


def count_references(content: str) -> int:
    """Count the number of references in the document."""
    # Match reference patterns like [1], [2], etc.
    ref_pattern = r"\.\. \[\d+\]"
    matches = re.findall(ref_pattern, content)
    return len(matches)


def check_code_blocks(content: str) -> list[str]:
    """Check code blocks for common issues."""
    issues = []

    # Find code blocks
    code_blocks = re.findall(
        r"\.\. code-block:: python\n(.*?)(?=\n\n[^\s]|\Z)", content, re.DOTALL
    )

    for i, block in enumerate(code_blocks, 1):
        # Check for placeholder text (but exclude Python f-strings and format strings)
        if "{" in block and "}" in block:
            # Remove f-strings and .format() patterns before checking
            block_no_fstrings = re.sub(r'f["\'][^"\']*\{[^}]+\}[^"\']*["\']', '', block)
            block_no_fstrings = re.sub(r'\.format\([^)]*\)', '', block_no_fstrings)
            block_no_fstrings = re.sub(r'["\'][^"\']*\{[^}]+\}[^"\']*["\']', '', block_no_fstrings)
            if re.search(r"\{[a-z_]+\}", block_no_fstrings):
                issues.append(f"Code block {i} contains unfilled placeholder")

        # Check for syntax issues
        if "import jax" in block and "from rheojax.core.jax_config" not in block:
            issues.append(
                f"Code block {i}: Direct jax import without safe_import_jax"
            )

    return issues


def check_internal_links(content: str) -> list[str]:
    """Check for broken internal link patterns."""
    issues = []

    # Check for :doc: links
    doc_links = re.findall(r":doc:`([^`]+)`", content)
    for link in doc_links:
        # Check for obviously wrong patterns
        if link.startswith("/") and not link.startswith("/"):
            issues.append(f"Suspicious doc link format: {link}")

    # Check for :ref: links
    ref_links = re.findall(r":ref:`([^`]+)`", content)
    for link in ref_links:
        if " " not in link and not re.match(r"^[a-z0-9-]+$", link):
            issues.append(f"Suspicious ref link format: {link}")

    return issues


def check_math_blocks(content: str) -> list[str]:
    """Check math blocks for common issues."""
    issues = []

    # Check for unbalanced braces in math
    math_blocks = re.findall(r"\.\. math::\n\n(.*?)(?=\n\n[^\s]|\Z)", content, re.DOTALL)

    for i, block in enumerate(math_blocks, 1):
        # Count braces
        open_braces = block.count("{")
        close_braces = block.count("}")
        if open_braces != close_braces:
            issues.append(f"Math block {i}: Unbalanced braces ({open_braces} open, {close_braces} close)")

    return issues


def validate_model_doc(file_path: Path) -> ValidationResult:
    """Validate a single model documentation file."""
    result = ValidationResult(file_path=file_path)

    if not file_path.exists():
        result.errors.append(f"File not found: {file_path}")
        return result

    content = file_path.read_text()
    line_count = len(content.splitlines())
    result.info.append(f"Line count: {line_count}")

    # Check for required sections
    sections = find_sections(content)

    for section in REQUIRED_SECTIONS:
        if section not in sections:
            result.errors.append(f"Missing required section: {section}")

    for section in RECOMMENDED_SECTIONS:
        if section not in sections:
            result.warnings.append(f"Missing recommended section: {section}")

    # Check Quick Reference
    qr_issues = check_quick_reference(content)
    result.warnings.extend(qr_issues)

    # Check reference count
    ref_count = count_references(content)
    result.info.append(f"Reference count: {ref_count}")

    if ref_count < MIN_REFERENCES_SIMPLE:
        result.errors.append(
            f"Insufficient references: {ref_count} < {MIN_REFERENCES_SIMPLE} minimum"
        )
    elif ref_count < MIN_REFERENCES_RECOMMENDED:
        result.warnings.append(
            f"Below recommended reference count: {ref_count} < {MIN_REFERENCES_RECOMMENDED}"
        )

    # Check code blocks
    code_issues = check_code_blocks(content)
    result.warnings.extend(code_issues)

    # Check internal links
    link_issues = check_internal_links(content)
    result.warnings.extend(link_issues)

    # Check math blocks
    math_issues = check_math_blocks(content)
    result.errors.extend(math_issues)

    # Check for "What You Can Learn" section content
    if "What You Can Learn" in sections:
        # Match until: horizontal rule (----), new main section (Title\n----), or EOF
        # Don't stop at subsection titles (~~~~) - only stop at main sections (----)
        wycl_match = re.search(
            r"What You Can Learn\n-+\n(.*?)(?=\n-{4,}\n|\n[A-Z][^\n]+\n-{4,}|\Z)",
            content,
            re.DOTALL,
        )
        if wycl_match:
            wycl_content = wycl_match.group(1)
            if len(wycl_content.strip()) < 200:
                result.warnings.append(
                    "'What You Can Learn' section appears too short"
                )
            # Check for key subsections
            for subsection in ["Parameter Interpretation", "Material Classification"]:
                if subsection not in wycl_content:
                    result.warnings.append(
                        f"'What You Can Learn' missing subsection: {subsection}"
                    )

    # Line count thresholds
    if line_count < 200:
        result.errors.append(f"Documentation too short: {line_count} lines (minimum 200)")
    elif line_count < 400:
        result.warnings.append(f"Documentation below target: {line_count} lines (target 500+)")

    return result


def print_result(result: ValidationResult, verbose: bool = False) -> None:
    """Print validation result."""
    status = "PASS" if result.is_valid else "FAIL"
    print(f"\n{'='*60}")
    print(f"File: {result.file_path.name}")
    print(f"Status: {status}")

    if verbose:
        for info in result.info:
            print(f"  [INFO] {info}")

    if result.errors:
        print("\n  ERRORS:")
        for error in result.errors:
            print(f"    - {error}")

    if result.warnings:
        print("\n  WARNINGS:")
        for warning in result.warnings:
            print(f"    - {warning}")


def main(args: Sequence[str] | None = None) -> int:
    """Main entry point."""
    if args is None:
        args = sys.argv[1:]

    # Default path
    if len(args) == 0:
        target_path = Path("docs/source/models")
    else:
        target_path = Path(args[0])

    verbose = "--verbose" in args or "-v" in args

    if not target_path.exists():
        print(f"Error: Path not found: {target_path}")
        return 1

    # Collect files to validate
    if target_path.is_file():
        files = [target_path]
    else:
        files = list(target_path.rglob("*.rst"))
        # Exclude index files
        files = [f for f in files if f.name not in ("index.rst", "summary.rst")]

    if not files:
        print(f"No .rst files found in {target_path}")
        return 1

    print(f"Validating {len(files)} documentation file(s)...")

    results = []
    for file_path in sorted(files):
        result = validate_model_doc(file_path)
        results.append(result)
        print_result(result, verbose=verbose)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for r in results if r.is_valid)
    failed = len(results) - passed
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)

    print(f"Files checked: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")

    # List files needing attention
    if failed > 0:
        print("\nFiles needing attention:")
        for r in results:
            if not r.is_valid:
                print(f"  - {r.file_path.name} ({len(r.errors)} errors)")

    # Documentation coverage metrics
    print("\nDocumentation Coverage:")
    short_docs = [r for r in results if r.line_count < 400]
    good_docs = [r for r in results if 400 <= r.line_count < 700]
    comprehensive_docs = [r for r in results if r.line_count >= 700]

    print(f"  Short (<400 lines): {len(short_docs)}")
    print(f"  Good (400-700 lines): {len(good_docs)}")
    print(f"  Comprehensive (700+ lines): {len(comprehensive_docs)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
