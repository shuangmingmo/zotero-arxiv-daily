"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import arxiv
import feedparser
import pytest

from zotero_arxiv_daily.retriever.arxiv_retriever import (
    ArxivRetriever,
    _run_with_hard_timeout,
    _split_rss_authors,
)
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def _hang(*args):
    time.sleep(30)


def _new_entries(feed):
    return [e for e in feed.entries if e.get("arxiv_announce_type", "new") == "new"]


def _patch_api(monkeypatch, results):
    """Replace arxiv.Client; `results(search)` is called for each API batch. Returns the list of searches."""
    searches = []

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            searches.append(search)
            return results(search)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    return searches


@pytest.fixture()
def no_downloads(monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)


def test_arxiv_retriever_builds_papers_from_rss_without_api(config, mock_feedparser, monkeypatch, no_downloads):
    def fail(search):
        raise AssertionError("arXiv API should not be called when RSS metadata is complete")

    _patch_api(monkeypatch, fail)
    new_entries = _new_entries(mock_feedparser)
    retriever = ArxivRetriever(config)
    assert retriever.run_notes() == []  # nothing to report before retrieval runs

    papers = retriever.retrieve_papers()

    assert [p.title for p in papers] == [e.title for e in new_entries]
    paper = papers[0]
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert paper.abstract.startswith("We propose a neural architecture search")
    assert paper.url == "https://arxiv.org/abs/2508.14001v1"
    assert paper.pdf_url == "https://arxiv.org/pdf/2508.14001v1"
    assert retriever.run_notes() == [
        "arXiv: 2 papers, metadata from the RSS feed; arXiv API not needed.",
        "arXiv full text: 0 of 2 papers.",
    ]
    assert retriever.subject_tags() == []


@pytest.mark.parametrize("status", [403, 406, 429, 503])
def test_arxiv_retriever_survives_blocked_api(config, mock_feedparser, monkeypatch, no_downloads, status):
    # Missing authors force an API lookup; the API then rejects the runner.
    new_entries = _new_entries(mock_feedparser)
    for entry in new_entries:
        entry["author"] = ""
    monkeypatch.setattr(arxiv_retriever, "API_BATCH_SIZE", 1)

    def blocked(search):
        raise arxiv.HTTPError(url="https://export.arxiv.org/api/query", retry=1, status=status)

    searches = _patch_api(monkeypatch, blocked)
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append, info=lambda msg: None))

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert [p.title for p in papers] == [e.title for e in new_entries]
    assert all(p.abstract for p in papers)
    assert len(searches) == 1  # the API is skipped for the rest of the run after the first block
    assert any(f"HTTP {status}" in w for w in warnings)
    assert f"arXiv API blocked (HTTP {status}) and skipped" in retriever.run_notes()[0]
    assert retriever.subject_tags() == ["API blocked"]


def test_arxiv_retriever_fills_missing_fields_from_api(config, mock_feedparser, monkeypatch, no_downloads):
    new_entries = _new_entries(mock_feedparser)
    new_entries[0]["summary"] = ""
    api_paper = arxiv.Result(
        entry_id="http://arxiv.org/abs/2508.14001v1",
        title="Title from API",
        authors=[arxiv.Result.Author("API Author")],
        summary="Abstract from API",
    )
    searches = _patch_api(monkeypatch, lambda search: iter([api_paper]))

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert searches[0].id_list == ["2508.14001v1"]
    assert papers[0].abstract == "Abstract from API"
    # Fields the RSS feed already provided are kept.
    assert papers[0].title == new_entries[0].title
    assert papers[0].authors == ["Alice Smith", "Bob Jones"]
    assert retriever.run_notes()[0] == (
        "arXiv: 2 papers, metadata from the RSS feed; arXiv API filled missing fields for 1 paper."
    )
    assert retriever.subject_tags() == []


def test_arxiv_retriever_uses_abstract_when_full_text_fails(config, mock_feedparser, monkeypatch, no_downloads):
    def tar_fails(paper):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", tar_fails)
    _patch_api(monkeypatch, lambda search: iter([]))

    papers = ArxivRetriever(config).retrieve_papers()

    assert len(papers) == len(_new_entries(mock_feedparser))
    paper = papers[0]
    assert paper.full_text is None

    prompts = []

    def create(**kwargs):
        prompts.append(kwargs["messages"][-1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A TLDR."))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm_params = {"api_mode": "chat_completion", "language": "English", "generation_kwargs": {"model": "m"}}
    assert paper.generate_tldr(client, llm_params) == "A TLDR."
    assert paper.abstract in prompts[0]


def test_arxiv_retriever_stops_full_text_after_consecutive_failures(config, mock_feedparser, monkeypatch, no_downloads):
    config.source.arxiv.include_cross_list = True  # all 7 fixture entries
    entries = mock_feedparser.entries
    _patch_api(monkeypatch, lambda search: iter([]))
    attempts = []

    def tar(paper):
        attempts.append(paper.title)
        # Only the third paper has full text; it resets the failure streak.
        return "full text" if paper.title == entries[2].title else None

    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", tar)
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append, info=lambda msg: None))

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    # fail, fail, success (reset), fail, fail, fail -> stop; the 7th paper is not downloaded.
    assert attempts == [e.title for e in entries[:6]]
    assert [p.full_text for p in papers] == [None, None, "full text", None, None, None, None]
    assert [p.title for p in papers] == [e.title for e in entries]
    assert all(p.abstract for p in papers)
    assert sum("rest of this run" in w for w in warnings) == 1
    assert retriever.run_notes()[1] == (
        "arXiv full text: 1 of 7 papers; downloads stopped after 3 papers in a row failed, the rest used abstracts."
    )
    assert retriever.subject_tags() == ["abstract-only"]


@pytest.mark.parametrize(
    "extractor, slow_step",
    [
        ("extract_text_from_tar", "_download_file"),
        ("extract_text_from_pdf", "_download_file"),
        ("extract_text_from_html", "_extract_text_from_html_worker"),
    ],
)
def test_full_text_download_is_time_limited(monkeypatch, extractor, slow_step):
    monkeypatch.setattr(arxiv_retriever, "FULL_TEXT_DOWNLOAD_TIMEOUT", 0.5)
    monkeypatch.setattr(arxiv_retriever, slow_step, _hang)
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    paper = arxiv.Result(
        entry_id="https://arxiv.org/abs/2508.14001v1",
        title="Slow paper",
        links=[arxiv.Result.Link("https://arxiv.org/pdf/2508.14001v1", title="pdf")],
    )

    start = time.monotonic()
    assert getattr(arxiv_retriever, extractor)(paper) is None
    assert time.monotonic() - start < 5
    assert "timed out" in warnings[0]


def test_split_rss_authors_drops_affiliations():
    creator = "Anthony Bertrand (UCA, LIMOS), Tom Schmitt (UCA),  Engelbert Mephu Nguifo"
    assert _split_rss_authors(creator) == ["Anthony Bertrand", "Tom Schmitt", "Engelbert Mephu Nguifo"]
    assert _split_rss_authors("") == []


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
