```markdown
# AGENTS.md

## Project Philosophy

This is a **learning project** about EEG signal processing and machine learning pipelines. The goal is to build a clear, understandable implementation of a brain-computer interface decoder, not a production system.

### Core Principles

#### KISS — Keep It Simple, Stupid

* Prefer straightforward implementations over clever abstractions
* Avoid premature optimization
* If a concept can be explained in one sentence, the code should reflect that
* When in doubt, choose the simpler solution
* No over-engineering: if a feature isn't required by the subject, don't add it

#### SOLID Principles

* **Single Responsibility**: Each module does one thing well
    * `loader.py` only loads data
    * `epoching.py` only creates epochs
    * `csp.py` only implements dimensionality reduction
* **Open/Closed**: Extend behavior without modifying existing code
    * New classifiers can be added without changing the pipeline structure
    * New preprocessing steps can be inserted without breaking existing flows
* **Liskov Substitution**: Implementations should be interchangeable
    * Custom transformers follow sklearn's interface exactly
* **Interface Segregation**: Keep interfaces minimal and focused
    * Functions accept only what they need
    * No "god objects" with dozens of parameters
* **Dependency Inversion**: Depend on abstractions, not concretions
    * Use sklearn's base classes for custom components
    * Pipeline components communicate through well-defined interfaces

---

## Code Style

### Typed Python

All functions must have type hints.

```python
# Good
def extract_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    tmin: float,
    tmax: float,
) -> tuple[mne.Epochs, np.ndarray]:
    ...

# Bad
def extract_epochs(raw, events, tmin, tmax):
    ...
```

Type hints serve as documentation and help catch errors early.

### Pydantic Models for Configuration and Data Structures

Use Pydantic models for:

* Configuration objects
* Data validation at boundaries
* Structured outputs

```python
from pydantic import BaseModel, Field

class EpochConfig(BaseModel):
    """Configuration for epoch extraction."""
    tmin: float = Field(default=-1.0, description="Start time before event (seconds)")
    tmax: float = Field(default=4.0, description="End time after event (seconds)")
    baseline: tuple[float, float] | None = Field(default=None, description="Baseline correction window")
    reject: float | None = Field(default=None, description="Peak-to-peak rejection threshold (microvolts)")

class SubjectScore(BaseModel):
    """Score result for a single subject."""
    subject_id: int
    experiment_type: str
    accuracy: float
    n_epochs: int
    n_train: int
    n_test: int
```

Pydantic provides:

* Automatic validation
* Clear error messages
* Schema documentation
* Easy serialization

### Validation Tools

* **ty**: Use for runtime type checking where needed
* **ruff**: Use for linting and formatting

Run ruff before commits:

```bash
ruff check tpv/
ruff format tpv/
```

Configure in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
```

---

## Documentation Standards

### Functions Must Be Documented

Every public function needs a docstring explaining:

* What it does
* Parameters
* Returns
* Raises (if applicable)

Use Google-style docstrings:

```python
def compute_csp_filters(
    epochs_data: np.ndarray,
    labels: np.ndarray,
    n_components: int = 4,
) -> np.ndarray:
    """Compute Common Spatial Patterns spatial filters.

    CSP finds spatial filters that maximize variance for one class
    while minimizing it for the other, making it effective for
    motor imagery EEG classification.

    Args:
        epochs_data: Array of shape (n_epochs, n_channels, n_times)
            containing the EEG epochs.
        labels: Array of shape (n_epochs,) with class labels (0 or 1).
        n_components: Number of spatial filters to retain.

    Returns:
        Projection matrix of shape (n_channels, n_components) where
        each column is a spatial filter.

    Raises:
        ValueError: If epochs_data has wrong shape or labels are invalid.
    """
    ...
```

### Modules Should Have Top-Level Documentation

```python
"""CSP (Common Spatial Patterns) implementation for EEG decoding.

This module provides a sklearn-compatible transformer that implements
the CSP algorithm for motor imagery classification. CSP is particularly
effective for distinguishing between different motor imagery tasks
by finding spatial filters that maximize class separability.

Example:
    >>> from tpv.csp import CSP
    >>> csp = CSP(n_components=4)
    >>> features = csp.fit_transform(epochs_data, labels)
"""
```

---

## Learning-Focused Design

### Keep Concepts Separate

Each concept should live in its own module:

* **Loading data** → `loader.py`
* **Understanding events** → `events.py`
* **Cleaning signals** → `preprocessing.py`
* **Creating training examples** → `epoching.py`
* **Feature extraction** → `csp.py`
* **Model assembly** → `pipeline.py`
* **Training** → `train.py`
* **Prediction** → `predict.py`
* **Evaluation** → `scoring.py`

This separation makes it easier to:

* Understand each step independently
* Debug issues in isolation
* Test components individually
* Explain the project to others

### Don't Overthink

This is a learning project. Prioritize:

* Clarity over performance
* Correctness over elegance
* Understanding over cleverness

If you're spending more than 30 minutes deciding between two designs, pick the simpler one.

### Avoid These Anti-Patterns

* **God files**: One file doing everything
* **Deep nesting**: More than 3 levels of indentation
* **Magic numbers**: Unexplained constants scattered in code
* **Implicit behavior**: Functions with hidden side effects
* **Copy-paste code**: Repeated logic instead of extracted functions

### Prefer These Patterns

* **Small functions**: Each function does one thing
* **Explicit is better than implicit**: Clear parameter names, no hidden state
* **Fail fast**: Validate inputs early, raise clear errors
* **Pure functions where possible**: Same input → same output, no side effects

---

## Example: How to Structure a New Feature

When adding a new feature, ask:

1. **Where does it belong?**
    * If it's about loading data → `loader.py`
    * If it's about signal processing → `preprocessing.py`
    * If it's about ML → `pipeline.py` or a new file

2. **What's the minimal interface?**
    * What inputs does it need?
    * What outputs does it produce?
    * Can it be a pure function?

3. **How do I validate it?**
    * Add type hints
    * Add a docstring
    * Add a simple test

4. **Is it understandable?**
    * Can you explain it in one sentence?
    * Would someone new to the project understand it?

---

## Testing Philosophy

Tests should be simple and focused.

```python
def test_csp_output_shape():
    """CSP should produce correct output shape."""
    # Arrange
    n_epochs, n_channels, n_times = 100, 64, 160
    epochs_data = np.random.randn(n_epochs, n_channels, n_times)
    labels = np.random.randint(0, 2, n_epochs)
    
    # Act
    csp = CSP(n_components=4)
    features = csp.fit_transform(epochs_data, labels)
    
    # Assert
    assert features.shape == (n_epochs, 4)
```

Test the important things:

* Output shapes
* Expected behavior on known inputs
* Edge cases (empty data, single class, etc.)

Don't test implementation details.

---

## Summary

* **Typed Python** with full type hints
* **Pydantic models** for configuration and structured data
* **ruff** for linting and formatting
* **KISS**: Simple solutions, clear code
* **SOLID**: Single responsibility, clear interfaces
* **Documented functions**: Every public function has a docstring
* **Learning-focused**: Clarity over cleverness

When in doubt, ask: "Would this be easy to explain to someone learning EEG processing?"

If the answer is no, simplify.
```