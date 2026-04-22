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

## Version bumping and changelog

**TODO list when releasing a new version:**
- [ ] Update `version` in `pyproject.toml`
- [ ] Update the `cff-version` in `CITATION.cff`
- [ ] Update the "Unreleased" section in `CHANGELOG.md` with the new version and release date, and move the changes from "Unreleased" to the new version section
- [ ] Add a link for the new version at the bottom of `CHANGELOG.md`
- [ ] Modify the "Unreleased" link at the bottom of `CHANGELOG.md` to point to the new version's commit range
- [ ] Run `uv lock` to update the lock file
- [ ] Push the changes and create a new release on GitHub with the new version tag