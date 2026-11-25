"""Tests for the store's dimension check. No DB: table dimensions are passed in."""

import pytest

from stockdesk.store import DimensionMismatch, check_dimension, filter_index_sql


def test_matching_or_new_table_passes():
    check_dimension("naive_1000", 1536, "text-embedding-3-small", 1536)
    check_dimension("naive_1000_large", None, "text-embedding-3-large", 3072)  # created on ingest


def test_mismatched_table_fails_clearly():
    with pytest.raises(DimensionMismatch, match="table naive_1000 stores 1536-dimensional vectors"):
        check_dimension("naive_1000", 1536, "text-embedding-3-large", 3072)


def test_filter_index_covers_doc_name_and_page():
    assert filter_index_sql("naive_1000_large") == (
        'CREATE INDEX IF NOT EXISTS "naive_1000_large_doc_name_page_idx" '
        'ON "naive_1000_large" (doc_name, page)'
    )


@pytest.mark.parametrize("table", ['x"; DROP TABLE y; --', "Naive", "1table", ""])
def test_filter_index_rejects_unsafe_table_names(table: str):
    with pytest.raises(ValueError, match="invalid TABLE name"):
        filter_index_sql(table)
