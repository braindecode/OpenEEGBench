# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add pytest to the CI workflow to run tests on each pull request ([#18](https://github.com/braindecode/OpenEEGBench/pull/18)).

### Changed
- not yet

### Fixed
- not yet


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

[Unreleased]: https://github.com/braindecode/OpenEEGBench/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/braindecode/OpenEEGBench/releases/tag/v0.2.1
[0.2.0]: https://github.com/braindecode/OpenEEGBench/releases/tag/v0.2.0
[0.1.0]: https://github.com/braindecode/OpenEEGBench/releases/tag/v0.1.0
