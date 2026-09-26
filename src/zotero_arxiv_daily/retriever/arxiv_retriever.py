from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
import multiprocessing
import os
import re
from queue import Empty
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
# Hard wall-clock limit for each full-text download (tar source, HTML page, PDF).
FULL_TEXT_DOWNLOAD_TIMEOUT = 20
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180
# After this many papers in a row without full text, stop downloading full text for the run.
FULL_TEXT_MAX_CONSECUTIVE_FAILURES = 3

API_BATCH_SIZE = 20
# Statuses meaning the arXiv API is refusing this client (e.g. HTTP 406 for GitHub-hosted
# runners). Once seen, the API is skipped for the rest of the run.
API_BLOCKED_STATUSES = {403, 406, 429, 503}
# RSS summaries look like "arXiv:2508.13426v1 Announce Type: new \nAbstract: <text>".
RSS_SUMMARY_HEADER = re.compile(r"^arXiv:\S+\s+Announce Type:\s*\S+\s*Abstract:\s*")


def _split_rss_authors(creator: str) -> list[str]:
    # dc:creator is "A, B (Affil, X), C": split on top-level commas and drop parenthesised affiliations.
    names, current, depth = [], "", 0
    for ch in creator:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif depth == 0:
            if ch == ",":
                names.append(current)
                current = ""
            else:
                current += ch
    names.append(current)
    return [" ".join(name.split()) for name in names if name.strip()]


def _rss_entry_to_result(entry) -> ArxivResult:
    arxiv_id = entry.id.removeprefix("oai:arXiv.org:")
    return ArxivResult(
        entry_id=f"https://arxiv.org/abs/{arxiv_id}",
        title=entry.get("title", "").strip(),
        authors=[ArxivResult.Author(name) for name in _split_rss_authors(entry.get("author", ""))],
        summary=RSS_SUMMARY_HEADER.sub("", entry.get("summary", "")).strip(),
        links=[ArxivResult.Link(f"https://arxiv.org/pdf/{arxiv_id}", title="pdf", rel="related", content_type="application/pdf")],
    )


def _missing_fields(paper: ArxivResult) -> list[str]:
    fields = {"title": paper.title, "abstract": paper.summary, "authors": paper.authors}
    return [name for name, value in fields.items() if not value]


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _download_file(url: str, path: str) -> str:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return path


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _download_with_timeout(url: str, path: str, paper_title: str) -> bool:
    return _run_with_hard_timeout(
        _download_file,
        (url, path),
        timeout=FULL_TEXT_DOWNLOAD_TIMEOUT,
        operation="Download",
        paper_title=paper_title,
    ) is not None


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(path: str, paper_id: str, paper_title: str | None = None) -> str | None:
    file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
    if not file_contents or "all" not in file_contents:
        raise ValueError("Main tex file not found.")
    return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
        # Run state, reported in the email via run_notes() and subject_tags().
        self._paper_count = None  # stays None until the RSS feed has been processed
        self._api_blocked_status = None
        self._api_filled = 0
        self._api_failed_requests = 0
        self._full_text_available = True
        self._full_text_failures = 0
        self._full_text_found = 0
        self._full_text_missing = 0

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        # The RSS feed carries all the metadata we need; the arXiv API is only used to fill gaps.
        raw_papers = [
            _rss_entry_to_result(i)
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            raw_papers = raw_papers[:10]

        self._fill_missing_from_api(raw_papers)

        usable_papers = []
        for paper in raw_papers:
            missing = _missing_fields(paper)
            if "title" in missing and "abstract" in missing:
                logger.warning(f"Skipping {paper.get_short_id()}: neither title nor abstract is available")
                continue
            if missing:
                logger.warning(f"{paper.get_short_id()} has no {', '.join(missing)}; continuing without it")
            usable_papers.append(paper)
        self._paper_count = len(usable_papers)
        return usable_papers

    def _fill_missing_from_api(self, raw_papers: list[ArxivResult]) -> None:
        incomplete = {p.get_short_id(): p for p in raw_papers if _missing_fields(p)}
        if not incomplete or self._api_blocked_status is not None:
            return
        logger.info(f"RSS metadata is incomplete for {len(incomplete)} papers, querying the arXiv API")
        client = arxiv.Client(num_retries=1, delay_seconds=3)
        paper_ids = list(incomplete)
        for i in range(0, len(paper_ids), API_BATCH_SIZE):
            search = arxiv.Search(id_list=paper_ids[i:i + API_BATCH_SIZE])
            try:
                api_papers = list(client.results(search))
            except arxiv.HTTPError as exc:
                if exc.status in API_BLOCKED_STATUSES:
                    logger.warning(f"arXiv API returned HTTP {exc.status}; skipping it for the rest of this run and using RSS metadata only")
                    self._api_blocked_status = exc.status
                    return
                logger.warning(f"arXiv API returned HTTP {exc.status} for batch {i // API_BATCH_SIZE}; using RSS metadata for it")
                self._api_failed_requests += 1
                continue
            except Exception as exc:
                logger.warning(f"arXiv API request failed for batch {i // API_BATCH_SIZE}: {exc}; using RSS metadata for it")
                self._api_failed_requests += 1
                continue
            for api_paper in api_papers:
                paper = incomplete.get(api_paper.get_short_id())
                if paper is None:
                    continue
                paper.title = paper.title or api_paper.title
                paper.summary = paper.summary or api_paper.summary
                paper.authors = paper.authors or api_paper.authors
                self._api_filled += 1

    def run_notes(self) -> list[str]:
        if self._paper_count is None:
            return []
        api_notes = []
        if self._api_filled:
            api_notes.append(f"arXiv API filled missing fields for {_count(self._api_filled, 'paper')}")
        if self._api_failed_requests:
            api_notes.append(f"arXiv API failed for {_count(self._api_failed_requests, 'request')}")
        if self._api_blocked_status is not None:
            api_notes.append(f"arXiv API blocked (HTTP {self._api_blocked_status}) and skipped")
        api = "; ".join(api_notes) or "arXiv API not needed"
        notes = [f"arXiv: {_count(self._paper_count, 'paper')}, metadata from the RSS feed; {api}."]
        total = self._full_text_found + self._full_text_missing
        if total:
            full_text = f"arXiv full text: {self._full_text_found} of {_count(total, 'paper')}"
            if not self._full_text_available:
                full_text += (
                    f"; downloads stopped after {FULL_TEXT_MAX_CONSECUTIVE_FAILURES} papers in a row failed, "
                    "the rest used abstracts"
                )
            notes.append(f"{full_text}.")
        return notes

    def subject_tags(self) -> list[str]:
        tags = []
        if self._api_blocked_status is not None:
            tags.append("API blocked")
        if not self._full_text_available:
            tags.append("abstract-only")
        return tags

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = self._extract_full_text(raw_paper) if self._full_text_available else None
        if full_text is None:
            self._full_text_missing += 1
        else:
            self._full_text_found += 1
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )

    def _extract_full_text(self, raw_paper: ArxivResult) -> str | None:
        # Full text is optional: without it, the TL;DR is generated from the abstract.
        for extract in (extract_text_from_tar, extract_text_from_html, extract_text_from_pdf):
            try:
                full_text = extract(raw_paper)
            except Exception as exc:
                logger.warning(f"Full-text extraction failed for {raw_paper.title}: {exc}")
                continue
            if full_text is not None:
                self._full_text_failures = 0
                return full_text
        self._full_text_failures += 1
        if self._full_text_failures >= FULL_TEXT_MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                f"No full text for {self._full_text_failures} papers in a row; "
                "skipping full-text downloads for the rest of this run and using abstracts instead"
            )
            self._full_text_available = False
        return None


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    # Fetching dominates; trafilatura's parse takes well under a second, so it shares the download limit.
    return _run_with_hard_timeout(
        _extract_text_from_html_worker,
        (html_url,),
        timeout=FULL_TEXT_DOWNLOAD_TIMEOUT,
        operation="HTML extraction",
        paper_title=paper.title,
    )


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        if not _download_with_timeout(paper.pdf_url, path, paper.title):
            return None
        return _run_with_hard_timeout(
            extract_markdown_from_pdf,
            (path,),
            timeout=PDF_EXTRACT_TIMEOUT,
            operation="PDF extraction",
            paper_title=paper.title,
        )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        if not _download_with_timeout(source_url, path, paper.title):
            return None
        return _run_with_hard_timeout(
            _extract_text_from_tar_worker,
            (path, paper.entry_id, paper.title),
            timeout=TAR_EXTRACT_TIMEOUT,
            operation="Tar extraction",
            paper_title=paper.title,
        )
