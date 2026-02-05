# Changelog

All changes we make to the assignment code or PDF will be documented in this file.

## [1.0.4] - 2025-05-19
### Changed
- code: Halve training tokens for the leaderboard run

## [1.0.3] - 2025-05-18

### Changed
- code: update Paloma validation set file name to `tokenized_paloma_c4_100_domains_validation.bin`, as it is a binary file loaded with `np.fromfile("/data/paloma/tokenized_paloma_c4_100_domains_validation.bin", dtype=np.uint16)`
- handout: add guidance to load the validation set with `np.fromfile`, and update references to new file name

## [1.0.2] - 2025-05-12

### Changed
- code: update `README.md` to clarify that students should use the provided training script, not their own train script
- code: update dependencies (`pyproject.toml` and `uv.lock`) with packages for WARC processing: `fastwarc` and `tldextract`
- handout: add hint to use `fastwarc` for WARC record iteration earlier in assignment
- handout: fix Together cluster paths to hatespeech and nsfw classifiers

## [1.0.1] - 2025-05-11

### Changed
- handout: clarify that students should use the provided training script, not their own train script
- handout: change references to WARC files in the final filtering step to WET files
- handout: provide hints on helpful classes to process the WET files

## [1.0.0] - 2025-05-07

### Added
- code: script to get all assets

### Changed
- code: improve supplied training script
- handout: update data to 2025
- handout: use WET files instead of WARC files for most tasks
- code: update deployment to use uv

## [0.0.4] - 2024-05-29

### Added

- handout: make sure to specify in problem `train_model` that we provide a training script.

### Changed

### Fixed

## [0.0.3] - 2024-05-26

### Added

### Changed

### Fixed

- handout: add `--device cuda` to training command

## [0.0.2] - 2024-05-19

### Added

- handout: added usage example for parallelism with `concurrent.futures` and `submitit`.
- handout: added points to each of the problems

### Changed

### Fixed

## [0.0.1] - 2024-05-14

### Added

### Changed

### Fixed

- code: fix type signature of `run_mask_emails`, `run_mask_phone_numbers`, and
  `run_mask_ips` adapters.
- code: fix expected labels in NSFW classifier test.
- handout: fix typo in mention of adapter `run_classify_quality` for problem
  `quality_classifier`.
- handout: fix link to Dolma NSFW and hatespeech classifiers, since the HF links
  point to the same model binary

## [0.0.0] - 2024-05-10

Initial release.
