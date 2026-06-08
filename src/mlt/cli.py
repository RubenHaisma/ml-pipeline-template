"""mlt — the binary. One Typer app, one subcommand per capability.

House rules (enforced in CI):
- every command takes ``--json``
- exit codes are load-bearing (0 ok, non-zero failure)
- no command writes state outside MLflow + ./artifacts
"""

from __future__ import annotations

import typer

from mlt import __version__
from mlt.commands.doctor import doctor
from mlt.commands.infer import infer
from mlt.commands.train import train

app = typer.Typer(
    name="mlt",
    help="CLI-first ML pipeline template — train, eval, serve, tracked in MLflow.",
    no_args_is_help=True,
    add_completion=False,
)

app.command()(doctor)
app.command()(train)
app.command()(infer)


@app.command()
def version(json_out: bool = typer.Option(False, "--json")) -> None:
    """Print the mlt version."""
    if json_out:
        typer.echo(f'{{"version": "{__version__}"}}')
    else:
        typer.echo(__version__)


if __name__ == "__main__":
    app()
