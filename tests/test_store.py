"""Tests for the store's dimension check. No DB: table dimensions are passed in."""

import pytest

from stockdesk.store import DimensionMismatch, check_dimension


def test_matching_or_new_table_passes():
    check_dimension("naive_1000", 1536, "text-embedding-3-small", 1536)
    check_dimension("naive_1000_large", None, "text-embedding-3-large", 3072)  # created on ingest


def test_mismatched_table_fails_clearly():
    with pytest.raises(DimensionMismatch, match="table naive_1000 stores 1536-dimensional vectors"):
        check_dimension("naive_1000", 1536, "text-embedding-3-large", 3072)
