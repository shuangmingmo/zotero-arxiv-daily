"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import pytest

from tests.canned_responses import make_sample_paper, make_stub_openai_client


@pytest.fixture()
def llm_params():
    return {
        "api_mode": "chat_completion",
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api_mode", ["chat_completion", "response"])
def test_tldr_returns_response(llm_params, api_mode):
    llm_params["api_mode"] = api_mode
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "Hello! How can I assist you today?"
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


def test_response_mode_maps_max_tokens(llm_params):
    from types import SimpleNamespace

    received_kwargs = {}

    def create_response(**kwargs):
        received_kwargs.update(kwargs)
        return SimpleNamespace(output_text="Summary")

    client = SimpleNamespace(
        responses=SimpleNamespace(create=create_response),
    )
    llm_params["api_mode"] = "response"
    paper = make_sample_paper()

    assert paper.generate_tldr(client, llm_params) == "Summary"
    assert received_kwargs["max_output_tokens"] == 16384
    assert "max_tokens" not in received_kwargs


def test_invalid_api_mode_falls_back_to_abstract(llm_params):
    llm_params["api_mode"] = "invalid"
    paper = make_sample_paper()

    assert paper.generate_tldr(make_stub_openai_client(), llm_params) == paper.abstract


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api_mode", ["chat_completion", "response"])
def test_affiliations_returns_parsed_list(llm_params, api_mode):
    llm_params["api_mode"] = api_mode
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def _recording_client(response: str):
    """Chat-completion client that returns `response` and records the messages it was sent."""
    from types import SimpleNamespace

    sent = []

    def create(**kwargs):
        sent.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), sent


def test_affiliations_keep_author_order_and_drop_repeats(llm_params):
    client, _ = _recording_client('["Univ B", "Univ A", "Univ B", " ", "Univ C"]')
    paper = make_sample_paper()
    assert paper.generate_affiliations(client, llm_params) == ["Univ B", "Univ A", "Univ C"]


def test_affiliations_prompt_starts_at_author_block_after_long_preamble(llm_params):
    # A preamble far longer than the prompt limit, including a macro *definition* named like an
    # author command, which must not be mistaken for the author block.
    preamble = "\\newcommand{\\address}[1]{#1}\n" + "".join(
        f"\\newcommand{{\\macro{i}}}{{\\mathcal{{O}}_{{{i}}}}}\n" for i in range(1500)
    )
    full_text = (
        "\\documentclass{article}\n" + preamble
        + "\\author{Aleksandr Artemev}\n"
        + "\\affiliation{Institute for Theoretical Physics, University of Amsterdam}\n"
        + "\\begin{document}\n\\maketitle\nBody.\n\\end{document}"
    )
    client, sent = _recording_client('["University of Amsterdam"]')
    paper = make_sample_paper(full_text=full_text)

    assert paper.generate_affiliations(client, llm_params) == ["University of Amsterdam"]
    system, user = sent[0]
    assert system["content"].startswith("You are an assistant who perfectly extracts affiliations")
    assert "\\affiliation{Institute for Theoretical Physics, University of Amsterdam}" in user["content"]
    assert "\\newcommand" not in user["content"]


@pytest.mark.parametrize(
    ("full_text", "expected_start"),
    [
        # JHEP style: \author[a]{...}
        ("\\usepackage{x}\n\\author[a]{A}\n\\affiliation[a]{U}", "\\author[a]{A}"),
        # amsart style: \address after \begin{document}
        ("\\newcommand{\\author}{}\n\\begin{document}\n\\address{U}", "\\address{U}"),
        # No author commands: start at the document body.
        ("\\usepackage{x}\n\\begin{document}\nText", "\\begin{document}"),
        # Text from HTML or PDF: unchanged.
        ("Title\nA. Author (University)\nAbstract ...", "Title"),
    ],
)
def test_affiliation_context_start(full_text, expected_start):
    from zotero_arxiv_daily.protocol import _affiliation_context

    assert _affiliation_context(full_text).startswith(expected_start)


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # re.search for [...] will fail -> AttributeError -> caught -> returns None
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None
