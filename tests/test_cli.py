import re

from polarpandas import __version__, cli


def test_cli_version_flag(capsys):
    exit_code = cli.main(["--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == __version__
    assert captured.err == ""


def test_cli_summary_flag(capsys):
    exit_code = cli.main(["--summary"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert re.search(r"PolarPandas", captured.out)
    assert "version" in captured.out.lower()


def test_cli_default_help(capsys):
    exit_code = cli.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "usage" in captured.err.lower()

