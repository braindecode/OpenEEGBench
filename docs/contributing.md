# Contributing

Contributions are welcome! Here are some ways to help:

- **Add a new backbone** — The [Braindecode](https://braindecode.org/) library always welcomes new backbone architectures! Follow the [model requirements](https://github.com/braindecode/braindecode/blob/master/CONTRIBUTING.md#adding-a-model-to-braindecode) and submit a PR to [braindecode's GitHub](https://github.com/braindecode/braindecode) with your backbone architecture. Then add a factory function in `default_configs/backbones.py` that loads your new model.
- **Add a new dataset** — Push a pre-windowed `BaseConcatDataset` to HuggingFace Hub and add a factory function in `default_configs/datasets.py`
- **Add a new fine-tuning strategy** — Implement a new class in `finetuning.py` inheriting from `BaseModel` with `apply()` and `get_callbacks()` methods
- **Report bugs or suggest features** — Open an issue on GitHub

## Development setup

```bash
uv pip install -e ".[dev]"
pytest  # make sure everything passes
```

## Running Tests

```bash
pytest                                # all tests
pytest tests/test_default_configs.py  # config validation
pytest tests/test_backbones.py        # model build & forward
pytest tests/test_normalization.py    # normalization correctness
pytest -k "test_name"                 # single test
```

## Releasing a new version

Releases follow [semantic versioning](https://semver.org/). Replace `X.Y.Z` below with the new version (e.g. `0.3.0`) and `YYYY-MM-DD` with today's date.

### 1. Open a release PR from a dedicated branch

```bash
git checkout main && git pull
git checkout -b release/X.Y.Z
```

### 2. Bump metadata

- [ ] `pyproject.toml` — set `version = "X.Y.Z"`.
- [ ] `CITATION.cff` — set `version: X.Y.Z` **and** `date-released: "YYYY-MM-DD"` (these are the software version and release date; **do not** touch `cff-version`, which is the CFF schema version).
- [ ] `CHANGELOG.md`:
  - Rename the existing `## [Unreleased]` header to `## [X.Y.Z] - YYYY-MM-DD`.
  - Insert a new empty `## [Unreleased]` section above it (required — every PR must update the changelog, so there must always be an Unreleased section).
  - At the bottom of the file, update the `[Unreleased]` link to `compare/vX.Y.Z...HEAD` and add a new `[X.Y.Z]: https://github.com/braindecode/OpenEEGBench/releases/tag/vX.Y.Z` line.

### 3. Refresh the lock file

```bash
uv lock
```

### 4. Commit, push, open the PR, wait for CI

```bash
git add -u && git commit -m "Version X.Y.Z"
git push -u origin release/X.Y.Z
gh pr create --base main --title "X.Y.Z" --body "Release X.Y.Z"
```

Wait for all checks to pass, then squash-merge into `main`.

### 5. Create the GitHub release

On [the releases page](https://github.com/braindecode/OpenEEGBench/releases/new):

- [ ] **Tag**: `vX.Y.Z` (with the `v` prefix — required by the `CHANGELOG.md` links, by previous tags, and by the Zenodo archive URL).
- [ ] **Target**: the squash-merge commit on `main`.
- [ ] **Title**: `vX.Y.Z`.
- [ ] **Description**: paste the `## [X.Y.Z]` section from `CHANGELOG.md`, followed by `**Full Changelog**: https://github.com/braindecode/OpenEEGBench/compare/vPREV...vX.Y.Z`.
- [ ] Click **Publish release** (not "Save draft" — the PyPI and Zenodo triggers only fire on `published`).

Clicking Publish automatically:
- runs [`publish.yml`](../.github/workflows/publish.yml) → uploads the sdist + wheels to PyPI and attaches them to the GitHub release;
- triggers the Zenodo webhook → archives the tagged snapshot and mints a new DOI under the concept DOI.

### 6. After publication

- [ ] Verify the package appears on [PyPI](https://pypi.org/project/open-eeg-bench/).
- [ ] Verify the release appears on [Zenodo](https://zenodo.org/) with the expected metadata (authors, affiliations, ORCIDs).
- [ ] On the first release only: replace the `TODO` placeholders in `README.md` (DOI badge and `doi` field in the BibTeX block) with the concept DOI from Zenodo, and open a follow-up PR.