"""Tests for domain.exceptions — hierarchy and catchability."""

import pytest

from seismic_visualizer.domain.exceptions import (
    AttributeNotApplicableError,
    EntityLimitReachedError,
    EntityNotFoundError,
    IndexOutOfBoundsError,
    LabelingNotAllowedError,
    SeismicError,
)

_CONCRETE_ERRORS = (
    IndexOutOfBoundsError,
    LabelingNotAllowedError,
    EntityNotFoundError,
    EntityLimitReachedError,
    AttributeNotApplicableError,
)


def test_all_are_subclass_of_seismic_error() -> None:
    for cls in _CONCRETE_ERRORS:
        assert issubclass(cls, SeismicError), f"{cls.__name__} must extend SeismicError"


def test_seismic_error_is_exception() -> None:
    assert issubclass(SeismicError, Exception)


@pytest.mark.parametrize("cls", _CONCRETE_ERRORS)
def test_catchable_as_seismic_error(cls: type[SeismicError]) -> None:
    with pytest.raises(SeismicError):
        raise cls("test message")


@pytest.mark.parametrize("cls", _CONCRETE_ERRORS)
def test_message_preserved(cls: type[SeismicError]) -> None:
    msg = "detailed error message"
    exc = cls(msg)
    assert str(exc) == msg
