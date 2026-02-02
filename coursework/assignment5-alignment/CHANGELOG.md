# Changelog

All changes we make to the assignment code or PDF will be documented in this file.

## [unreleased] - yyyy-mm-dd

### Added

### Changed

### Fixed

## [1.0.2] - 2025-06-01

### Added
- handout: added guidance on size of D_b in expert iteration experiment
- code: add data for optional assignment

### Changed

### Fixed

## [1.0.1] - 2025-05-31

### Added
- code: for optional assignment, update alpaca_eval to Llama 3.3 70B Instruct judge
- handout: add optional assignment on safety, instruction tuning, and RLHF

### Changed

### Fixed
- code: change masked normalize constant to not equal seqlen, more SFT test coverage

## [1.0.0] - 2025-05-23

### Added
- code: 2025 assignment on SFT, Expert Iteration, and GRPO with verified rewards on MATH
- handout: 2025 assignment on SFT, Expert Iteration, and GRPO with verified rewards on MATH

### Changed
- N/A

### Fixed
- N/A

## [0.0.3] - 2024-06-05

### Added

### Changed

- code: fix AlpacaEval auto-evaluator to use local Llama 3 70B instruct.
- code: add missing `evaluate_safety.py` script to `scripts`
- code: add Llama 3 tokenizer as a fixture.
- code: fix DPO loss test
- code: make SFT dataset test stricter by comparing against expected output to help folks catch bugs.
- code: include prompts as text files in `cs336_alignment/prompts`
- handout: fix typo in code example for writing AlpacaEval outputs.
- handout: provide more instructions on interpreting AlpacaEval annotations file.
- handout: give better default DPO hyperparameters
- handout: clarify prompt to use for the DPO loss (AlpacaEval prompt) and mention EOS token
- handout: clarify that arrows in the prompts are line continuations, not line breaks
- handout: mention that we provide the prompts as text files at `cs336_alignment/prompts`

### Fixed

## 0.0.2 - 2024-05-30

### Added

- code: add MMLU, GSM8K, AlpacaEval, and SimpleSafetyTests data to `./data`.

### Changed

### Fixed


## 0.0.1 - 2024-05-30

### Added

- handout: explicitly set CUDA_HOME in FlashAttention-2 installation instructions. 
- code: explicitly set CUDA_HOME in FlashAttention-2 installation instructions.

### Changed

### Fixed


## 0.0.0 - 2024-05-30

Initial release
