# Contributing to pbg-pennylane-data-reuploading

## Development setup

uv is required. Install with `brew install uv` or `pip install uv`.

    uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
    pytest

## Releasing to PyPI

Tag a commit with `git tag v<VERSION>` and push the tag. The
`.github/workflows/release.yml` workflow publishes to PyPI automatically
using trusted publishing (no tokens needed after initial setup).

PyPI trusted publishing must be configured once per repo. See
https://docs.pypi.org/trusted-publishers/ and
[docs/conventions/distribution.md](https://github.com/vivarium-collective/pbg-superpowers/blob/main/docs/conventions/distribution.md).
