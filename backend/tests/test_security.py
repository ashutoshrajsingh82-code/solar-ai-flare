
import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_cors_rejects_wildcard_origin():
    with pytest.raises(ValidationError):
        Settings(cors_origins="*")


def test_cors_rejects_wildcard_among_multiple_origins():
    with pytest.raises(ValidationError):
        Settings(cors_origins="https://example.com,*")


def test_cors_parses_multiple_origins():
    settings = Settings(
        cors_origins=" https://example.com, https://app.example.com "
    )

    assert settings.allowed_origins == [
        "https://example.com",
        "https://app.example.com",
    ]


def test_cors_empty_origins_allow_no_origins():
    settings = Settings(cors_origins="")

    assert settings.allowed_origins == []


def test_cors_ignores_empty_entries():
    settings = Settings(
        cors_origins="https://example.com,, https://app.example.com,"
    )

    assert settings.allowed_origins == [
        "https://example.com",
        "https://app.example.com",
    ]
