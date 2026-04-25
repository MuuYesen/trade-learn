"""Tests for Alpha formula metadata helpers."""

from tradelearn.factor.alpha import (
    ALPHA101_SKIPPED,
    ALPHA101_SUPPORTED,
    ALPHA191_SKIPPED,
    ALPHA191_SUPPORTED,
)


def test_alpha_formula_metadata_lists_supported_and_skipped_formulas() -> None:
    """Callers can discover supported and intentionally skipped Alpha formulas."""
    from tradelearn.factor import alpha as alpha_package

    alpha_formula_metadata = alpha_package.alpha_formula_metadata
    metadata = alpha_formula_metadata()

    assert metadata == {
        "alpha101": {
            "supported": tuple(sorted(ALPHA101_SUPPORTED)),
            "skipped": ALPHA101_SKIPPED,
        },
        "alpha191": {
            "supported": tuple(sorted(ALPHA191_SUPPORTED)),
            "skipped": ALPHA191_SKIPPED,
        },
    }


def test_alpha_formula_metadata_returns_skipped_copies() -> None:
    """Mutating metadata from one call must not change the package constants."""
    from tradelearn.factor import alpha as alpha_package

    alpha_formula_metadata = alpha_package.alpha_formula_metadata
    metadata = alpha_formula_metadata()

    metadata["alpha101"]["skipped"]["alpha999"] = "local mutation"
    metadata["alpha191"]["skipped"]["alpha999"] = "local mutation"

    fresh = alpha_formula_metadata()
    assert "alpha999" not in ALPHA101_SKIPPED
    assert "alpha999" not in ALPHA191_SKIPPED
    assert "alpha999" not in fresh["alpha101"]["skipped"]
    assert "alpha999" not in fresh["alpha191"]["skipped"]


def test_alpha_formula_metadata_is_exported_from_package_all() -> None:
    """The helper is part of the public alpha facade."""
    import tradelearn.factor.alpha as alpha_package

    assert "alpha_formula_metadata" in alpha_package.__all__


def test_alpha_formula_metadata_is_exported_from_factor_package() -> None:
    """The top-level factor facade exposes Alpha formula metadata."""
    import tradelearn.factor as factor_package
    import tradelearn.factor.alpha as alpha_package

    assert factor_package.alpha_formula_metadata() == alpha_package.alpha_formula_metadata()
    assert "alpha_formula_metadata" in factor_package.__all__


def test_alpha_formula_constants_are_exported_from_factor_package() -> None:
    """The top-level factor facade exposes Alpha formula metadata constants."""
    import tradelearn.factor as factor_package
    import tradelearn.factor.alpha as alpha_package

    assert factor_package.ALPHA101_SUPPORTED == alpha_package.ALPHA101_SUPPORTED
    assert factor_package.ALPHA191_SUPPORTED == alpha_package.ALPHA191_SUPPORTED
    assert factor_package.ALPHA101_SKIPPED == alpha_package.ALPHA101_SKIPPED
    assert factor_package.ALPHA191_SKIPPED == alpha_package.ALPHA191_SKIPPED
    assert {
        "ALPHA101_SKIPPED",
        "ALPHA101_SUPPORTED",
        "ALPHA191_SKIPPED",
        "ALPHA191_SUPPORTED",
    }.issubset(factor_package.__all__)


def test_alpha_formula_facades_are_exported_from_factor_package() -> None:
    """The top-level factor facade exposes both Alpha formula callable facades."""
    import tradelearn.factor as factor_package
    import tradelearn.factor.alpha as alpha_package

    assert factor_package.alpha101 is alpha_package.alpha101
    assert factor_package.alpha191 is alpha_package.alpha191
    assert {"alpha101", "alpha191"}.issubset(factor_package.__all__)
