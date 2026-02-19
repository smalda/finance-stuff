# Shared Library Style Guide

Rules for all Python modules in `course/shared/`. Consumed by humans,
week-code agents, and `generate_api.py`.

---

## 1. Docstrings — Google Style

Every public function and class uses Google-style docstrings. No NumPy-style.

```python
def foo(x: np.ndarray, n: int = 10) -> dict:
    """One-line summary ending with a period.

    Optional longer description. Keep it concise.

    Args:
        x: description (no type repetition — types are in the signature).
        n: description.

    Returns:
        dict with keys:
            key_a: float — description.
            key_b: bool — description.

    Raises:
        ValueError: when n < 1.
    """
```

**Rules:**
- First line = one-sentence summary, imperative mood, ending with period.
- `Args:` block if 1+ parameters (skip `self`). No type in the description
  (it's already in the signature).
- `Returns:` block always present for non-`None` returns.
  - **Dict returns must enumerate keys.** This is the most important rule.
    Downstream agents cannot read source — they only see `generate_api.py`
    output. `-> dict` with no key list is useless.
  - Tuple returns should name each element.
- `Yields:` instead of `Returns:` for generators.
- `Raises:` block if the function raises on invalid input.
- Classes: docstring on the class, not on `__init__`. `Args:` in the class
  docstring covers constructor parameters.

---

## 2. Type Annotations

Every public function signature has full annotations (parameters + return).

```python
# Functions
def pearson_ic(predicted: np.ndarray, actual: np.ndarray) -> float:

# Generators
def walk_forward_splits(
    dates: np.ndarray, train_window: int, purge_gap: int = 1,
) -> Iterator[tuple[np.ndarray, Any]]:

# Dict returns — use plain dict, document keys in docstring
def ic_summary(ic_series: np.ndarray) -> dict:

# Classes — annotate __init__ params, split/get_n_splits return types
def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
```

**Rules:**
- All files already have `from __future__ import annotations`. Do NOT use
  string forward references (`"np.ndarray"`) — rely on the import instead.
- Use `X | Y` union syntax, not `Optional[X]` or `Union[X, Y]`.
- Generators: annotate as `-> Iterator[...]` (import from `collections.abc`).
- Private helpers (`_foo`): annotations encouraged but not required.
- Math-notation single-letter params (`S`, `K`, `T`, `r`, `sigma`) are
  acceptable in derivatives/portfolio where they match textbook convention.

---

## 3. Error Handling Contract

Three categories, documented in each module's docstring or per-function:

| Category | Contract | Example modules |
|---|---|---|
| **Metric functions** | Return `np.nan` on degenerate input (too few obs, zero variance). Never raise on bad data. | metrics, backtesting (perf metrics) |
| **Config validators** | Raise `ValueError` immediately on invalid configuration. | temporal (CV splitters), dl_training |
| **Data loaders** | Raise on missing cache / network failure. Print warnings for partial data. | data |
| **Stubs** | Raise `NotImplementedError` with implementation hint. | nlp, causal, rl_env, microstructure (some) |

**Rules:**
- Metric functions: guard clause at top, return `np.nan`. No exceptions.
- CV splitters: validate `n_splits`, data length in `split()`. Raise `ValueError`.
- Document error behavior in `Raises:` or `Returns:` docstring.
- Never silently swallow exceptions (catch + continue) without logging.

---

## 4. Naming

- **Functions and variables**: `snake_case`.
- **Classes**: `PascalCase`.
- **Constants**: `UPPER_SNAKE_CASE`.
- **Math variables**: single-letter allowed in derivatives/portfolio modules
  where they match standard notation (`S` = spot, `K` = strike, `T` = time,
  `r` = rate, `sigma` = vol, `mu` = expected return, `cov` = covariance).
- **No abbreviations** in non-math contexts. Use `ticker_str` not `tk`.

---

## 5. Module Structure

```python
"""Module-level docstring: one-line summary.

Longer description. Which weeks use this. External deps.
"""
from __future__ import annotations

# stdlib
from collections.abc import Iterator

# third-party
import numpy as np

# local
from .metrics import pearson_ic
```

**Rules:**
- `from __future__ import annotations` in every file (already true).
- Lazy imports for heavy/optional deps (torch, hmmlearn, transformers).
- Section separators (`# ---...`) for logical groupings within a module.

---

## 6. generate_api.py Output Requirements

The API.md that `generate_api.py` produces is consumed by Step 4A agents.
It must be self-describing:

- **Dict return keys** are extracted from the `Returns:` docstring block
  and rendered inline.
- **Constants** are grouped by category (not one long line).
- **Generator return types** show `Iterator[...]` in the signature line.
- **Error contract** is visible per-function (raises vs returns NaN).
