"""Output formatting utilities for the RheoJAX CLI.

All human-readable output should go through the helpers in this module so
that ``--no-color`` and ``--json`` flags are respected consistently.
Progress spinners and status messages are written to stderr so that stdout
remains clean for JSON envelope piping.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.theme import Theme

from rheojax.logging import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Console singleton
# ------------------------------------------------------------------

_console: Console | None = None

_RHEOJAX_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "header": "bold blue",
        "muted": "dim white",
    }
)


def get_console(no_color: bool = False) -> Console:
    """Return the module-level :class:`rich.console.Console` singleton.

    The console writes to *stderr* so stdout stays available for JSON
    envelope piping.  The singleton is re-created if ``no_color`` changes.

    Args:
        no_color: When ``True`` strips all ANSI escape codes from output.

    Returns:
        Shared :class:`rich.console.Console` instance.

    Example:
        >>> console = get_console(no_color=True)
    """
    global _console

    if _console is None:
        _console = Console(
            stderr=True,
            theme=_RHEOJAX_THEME,
            no_color=no_color,
            highlight=not no_color,
        )
    return _console


def reset_console(no_color: bool = False) -> Console:
    """Force-recreate the console singleton (useful after parsing --no-color).

    Args:
        no_color: Whether to disable color.

    Returns:
        New :class:`rich.console.Console` instance.
    """
    global _console
    _console = Console(
        stderr=True,
        theme=_RHEOJAX_THEME,
        no_color=no_color,
        highlight=not no_color,
    )
    return _console


# ------------------------------------------------------------------
# Formatted output helpers
# ------------------------------------------------------------------


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    no_color: bool = False,
    **kwargs: Any,
) -> None:
    """Print a formatted table to stderr.

    Args:
        title: Table caption displayed above the grid.
        headers: Column header labels.
        rows: Row data — each inner list must have the same length as *headers*.
        no_color: Forward to :func:`get_console` if console not yet created.
        **kwargs: Forwarded to :class:`rich.table.Table` constructor.

    Example:
        >>> print_table("Models", ["Name", "Protocol"], [["maxwell", "relaxation"]])
    """
    console = get_console(no_color=no_color)
    table = Table(title=title, **kwargs)
    for header in headers:
        table.add_column(header, style="info")
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)


def print_result(title: str, items: dict[str, str], no_color: bool = False) -> None:
    """Print a key-value result block inside a bordered panel.

    Args:
        title: Panel heading.
        items: Ordered mapping of label -> value strings.
        no_color: Forward to :func:`get_console`.

    Example:
        >>> print_result("Fit Result", {"G_e": "1000.0 Pa", "tau": "0.1 s"})
    """
    console = get_console(no_color=no_color)
    lines: list[str] = []
    key_width = max((len(k) for k in items), default=10)
    for key, value in items.items():
        lines.append(f"[header]{key:<{key_width}}[/header]  {value}")
    body = "\n".join(lines)
    console.print(Panel(body, title=f"[header]{title}[/header]", expand=False))


def print_error(msg: str, no_color: bool = False) -> None:
    """Print an error message to stderr in red.

    Args:
        msg: Error description.
        no_color: Forward to :func:`get_console`.

    Example:
        >>> print_error("File not found: data.csv")
    """
    console = get_console(no_color=no_color)
    console.print(f"[error]Error:[/error] {msg}")


def print_warning(msg: str, no_color: bool = False) -> None:
    """Print a warning message to stderr in yellow.

    Args:
        msg: Warning description.
        no_color: Forward to :func:`get_console`.

    Example:
        >>> print_warning("No convergence reached; using best-effort result.")
    """
    console = get_console(no_color=no_color)
    console.print(f"[warning]Warning:[/warning] {msg}")


def print_success(msg: str, no_color: bool = False) -> None:
    """Print a success message to stderr in green.

    Args:
        msg: Success description.
        no_color: Forward to :func:`get_console`.

    Example:
        >>> print_success("Pipeline completed in 3.2 s.")
    """
    console = get_console(no_color=no_color)
    console.print(f"[success]✓[/success] {msg}")


def create_progress(no_color: bool = False) -> Progress:
    """Create a :class:`rich.progress.Progress` context for long operations.

    The progress bar writes to stderr so JSON output on stdout is not
    polluted.

    Args:
        no_color: Whether to strip colors.

    Returns:
        Configured :class:`rich.progress.Progress` instance.

    Example:
        >>> with create_progress() as progress:
        ...     task = progress.add_task("Fitting...", total=None)
    """
    console = get_console(no_color=no_color)
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
