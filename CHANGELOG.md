# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `ScratchBackbone` for benchmarking DL models without pretrained weights. Only compatible with `FullFinetune` (enforced by the `Experiment` validator) ([#30](https://github.com/braindecode/OpenEEGBench/pull/30)).
- Add `from_scratch=True` flag to `benchmark()` for one-call scratch benchmarks; defaults `finetuning_strategies` to `["full_finetune"]` and rejects pretrained-weight inputs. New `examples/benchmark_from_scratch.py` runs the full braindecode classifier suite from scratch on every dataset, batched into a single SLURM job array.
- Add `class_weight` parameter to `RidgeProbingTraining` (`"balanced"` or `None`); **default changed to `"balanced"`** — pass `None` for the previous unweighted behavior ([#32](https://github.com/braindecode/OpenEEGBench/pull/32)).
- Add `dtype` parameter to `RidgeProbingTraining` (`"float32"` or `"float64"`, default `"float64"`). Use `"float32"` only when necessary, e.g. on Apple MPS which does not support float64 ([#32](https://github.com/braindecode/OpenEEGBench/pull/32)).

### Changed
- Fill in the Zenodo concept DOI (`10.5281/zenodo.19698863`) in the README DOI badge and the BibTeX snippet, and add it as an `identifiers` entry in `CITATION.cff`.


## [0.3.0] - 2026-04-22

### Added
- Add `"ridge_probe"` finetuning strategy: closed-form streaming ridge regression probing on frozen backbone features. Single pass over the dataloader, eigendecomposition-based λ sweep, no hyperparameter tuning needed. For high-dimensional backbones, `max_features` (default 5000) triggers a Gaussian random projection seeded by `Experiment.seed`; multi-seed runs therefore produce different projections and enable variance estimation ([#21](https://github.com/braindecode/OpenEEGBench/pull/21)).
- Add `FlattenHead` head type (used internally by ridge probing) ([#21](https://github.com/braindecode/OpenEEGBench/pull/21)).
- Add `RidgeProbingTraining` config with `kind="ridge"` discriminator for the `Training` union ([#21](https://github.com/braindecode/OpenEEGBench/pull/21)).
- Add pytest to the CI workflow to run tests on each pull request ([#18](https://github.com/braindecode/OpenEEGBench/pull/18)).
- Add `max_meta_experiments` argument to `helpers.run_multiple_per_node()` as alternative to `max_experiments_per_node` ([#17](https://github.com/braindecode/OpenEEGBench/pull/17)).
- Add `training_required_modules` parameter to the backbones ([#20](https://github.com/braindecode/OpenEEGBench/pull/20))
- Expose the `preload` parameter of datasets [(#26)](htttps://github.com/braindecode/OpenEEGBench/pull/26).
- Add `CITATION.cff` metadata file and a Zenodo DOI badge in the README for archival citation via Zenodo.

### Changed
- Popularize the use of `import open_eeg_bench as oeb` via the README and documentation ([#17](https://github.com/braindecode/OpenEEGBench/pull/17)).
- Improve import hints via the `__all__` variable in `__init__.py` ([#17](https://github.com/braindecode/OpenEEGBench/pull/17)).
- All backbones now use `peft_target_modules="all-linear"` by default for simplicity, which leads to a slight increase in the number of parameters being finetuned ([#20](https://github.com/braindecode/OpenEEGBench/pull/20)).
- Change default `LoRA.bias` to "none" to match PEFT's default ([#20](https://github.com/braindecode/OpenEEGBench/pull/20)).
- Allow disabling dropout layers of the backbone (default: True) ([#20](https://github.com/braindecode/OpenEEGBench/pull/20)).
- Change default `Experiment.seed` from 42 to 0 ([#21](https://github.com/braindecode/OpenEEGBench/pull/21)).

### Fixed
- The lazy modules are now initialized before and after applying the PEFT (necessary for some PEFT methods like OFT) ([#17](https://github.com/braindecode/OpenEEGBench/pull/17)).
- Take into account [BENDR's channels projection fix](https://github.com/braindecode/braindecode/pull/954) in the BENDR backbone ([#20](https://github.com/braindecode/OpenEEGBench/pull/20)).


## [0.2.1] - 2026-04-07

### Added
- PyPI metadata: project URLs, and classifiers ([#14](https://github.com/braindecode/OpenEEGBench/pull/14)).
- And README badges (PyPI version, supported Python versions, tests status, HuggingFace Space) ([#14](https://github.com/braindecode/OpenEEGBench/pull/14)).
- Add step in publish.yml workflow to publish the build artifacts on the GitHub release ([#14](https://github.com/braindecode/OpenEEGBench/pull/14)).


## [0.2.0] - 2026-04-07

### Added
- Benchopt interface for running benchmarks via the `benchopt_wrappers` directory ([#8](https://github.com/braindecode/OpenEEGBench/pull/8)).
- Experiment wrapper allowing multiple experiments to run on the same node via Joblib ([#10](https://github.com/braindecode/OpenEEGBench/pull/10)).
- `NoNormalization` transform to simplify serialization ([#11](https://github.com/braindecode/OpenEEGBench/pull/11)).
- Option for `collect_results` to wait for pending jobs ([#11](https://github.com/braindecode/OpenEEGBench/pull/11)).
- Experimental warnings on helper functions ([#10](https://github.com/braindecode/OpenEEGBench/pull/10)).
- Skip experiments that are already completed ([#10](https://github.com/braindecode/OpenEEGBench/pull/10)).
- Optional sorting in result collection ([#10](https://github.com/braindecode/OpenEEGBench/pull/10)).

### Changed
- Bumped `braindecode` dependency to 1.4.0 ([#12](https://github.com/braindecode/OpenEEGBench/pull/12)).
- Set dataset `preload=False` by default ([#11](https://github.com/braindecode/OpenEEGBench/pull/11)).
- Improved error parsing and infra `model_dump` output ([#10](https://github.com/braindecode/OpenEEGBench/pull/10)).
- Documentation now uses the `pip install` command for installation instructions (382e833).

### Fixed
- Corrected EEG montages handling ([#9](https://github.com/braindecode/OpenEEGBench/pull/9)).
- Removed `infra` from `_exclude_from_cls_uid` so jobs correctly honor the `retry` and `force` modes ([#10](https://github.com/braindecode/OpenEEGBench/pull/10)).


## [0.1.0] - 2026-03-31

Initial release.

[Unreleased]: https://github.com/braindecode/OpenEEGBench/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/braindecode/OpenEEGBench/releases/tag/v0.3.0
[0.2.1]: https://github.com/braindecode/OpenEEGBench/releases/tag/v0.2.1
[0.2.0]: https://github.com/braindecode/OpenEEGBench/releases/tag/v0.2.0
[0.1.0]: https://github.com/braindecode/OpenEEGBench/releases/tag/v0.1.0
