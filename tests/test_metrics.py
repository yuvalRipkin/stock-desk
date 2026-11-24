from stockdesk.evals.metrics import distinct_pages, evidence_hit, is_refusal, page_hit

EVIDENCE = (
    "Net revenue for 2022 was $23.6 billion, an increase of 44% compared to 2021. "
    "The increase was driven by Embedded and Data Center segment growth."
)


# page_hit


def test_page_hit_hit():
    assert page_hit([3, 47, 12], [47])


def test_page_hit_miss():
    assert not page_hit([3, 12], [47, 48])


def test_page_hit_empty_retrieved():
    assert not page_hit([], [47])


def test_page_hit_empty_evidence():
    assert not page_hit([47], [])


# evidence_hit


def test_evidence_hit_ignores_case_and_whitespace():
    chunk = "NET   REVENUE for 2022\nwas $23.6 billion,\tan increase of 44% compared to 2021."
    assert evidence_hit([chunk], ["Net revenue for 2022 was $23.6 billion, an increase of 44%"])


def test_evidence_hit_unrelated_text():
    chunk = "The company leases office space in Santa Clara, California and Austin, Texas."
    assert not evidence_hit([chunk], [EVIDENCE])


def test_evidence_hit_evidence_longer_than_chunk():
    chunk = EVIDENCE[20:90]
    assert len(chunk) < len(EVIDENCE)
    assert evidence_hit([chunk], [EVIDENCE])


def test_evidence_hit_empty_chunks():
    assert not evidence_hit([], [EVIDENCE])


# is_refusal


def test_is_refusal_exact():
    assert is_refusal("Not found in the filing.")


def test_is_refusal_case_insensitive():
    assert is_refusal("NOT FOUND IN THE FILING")


def test_is_refusal_ignores_whitespace_and_end_punctuation():
    assert is_refusal("  Not found\nin the filing!  ")
    assert is_refusal('"Not found in the filing."') is False  # judge grades it instead


def test_hedged_answer_is_not_a_refusal():
    answer = "Customer deposits were the largest liability in 2022. Not found in the filing."
    assert not is_refusal(answer)


def test_refusal_followed_by_explanation_is_not_a_refusal():
    assert not is_refusal("Not found in the filing. The excerpts do not include the balance sheet.")


def test_phrase_inside_a_real_answer_is_not_a_refusal():
    assert not is_refusal("I checked the excerpts, but this is not found in the filing, sorry.")
    assert not is_refusal("0 (Not found in the filing.)")


def test_is_refusal_normal_answer():
    assert not is_refusal("Total net revenue in 2022 was $23.601 billion (page 47).")


# distinct_pages


def test_distinct_pages_with_duplicates():
    assert distinct_pages([47, 47, 12, 3, 12]) == 3


def test_distinct_pages_empty():
    assert distinct_pages([]) == 0
