"""Tests for zotero_arxiv_daily.text_format: LaTeX names, titles and LLM summaries rendered for email."""

import pytest

from zotero_arxiv_daily.text_format import latex_accents_to_unicode, latex_to_unicode, tex_to_html


@pytest.mark.parametrize(
    ("tex", "expected"),
    [
        (r"Bergische Universit\"at Wuppertal", "Bergische Universität Wuppertal"),
        (r"Universit\"{a}t", "Universität"),
        (r"Physique Th\'eorique", "Physique Théorique"),
        (r"Ji\v{r}\'i", "Jiří"),
        (r"Ba\u{g}c{\i}", "Bağcı"),
        (r"Fran\c{c}ois and \c c", "François and ç"),
        (r"{\O}ystein, Pawe\l{} and {\aa}", "Øystein, Paweł and å"),
        (r"Gau\ss stra\ss e 20", "Gaußstraße 20"),
        (r"Mar\'{\i}a and Nu\~nez", "María and Nuñez"),
    ],
)
def test_latex_accents_to_unicode_converts_accents(tex, expected):
    assert latex_accents_to_unicode(tex) == expected


def test_latex_accents_to_unicode_keeps_other_markup():
    tex = r"\author[a]{A}\affiliation[a]{B} \vec{x} \label{y} \textit{z} \iota \omega \ldots \cite{w} \% \&"
    assert latex_accents_to_unicode(tex) == tex


@pytest.mark.parametrize(
    ("tex", "expected"),
    [
        (r"Ji\v{r}\'i Novotn\'y", "Jiří Novotný"),
        (r"David Podiv\'in", "David Podivín"),
        (r"Ali Ba\u{g}c{\i}", "Ali Bağcı"),
        (r"Z\"uhra G\"une\c{s}", "Zühra Güneş"),
        (r"{\O}ystein S\o{}rensen", "Øystein Sørensen"),
        (r"Arkadiusz Or{\l}owski", "Arkadiusz Orłowski"),
        (r"Schr\"odinger equation for {QCD}", "Schrödinger equation for QCD"),
        ("Alice Smith", "Alice Smith"),
    ],
)
def test_latex_to_unicode(tex, expected):
    assert latex_to_unicode(tex) == expected


def test_latex_to_unicode_leaves_prose_characters_alone():
    # In LaTeX, % starts a comment and & is a column separator; in prose they are text.
    assert latex_to_unicode(r"50% of A & B at Universit\'e") == "50% of A & B at Université"


def test_tldr_from_screenshot_renders_bold_and_math():
    tldr = (
        r"The paper derives new **quadratic sum rules** within its string-inspired \(\alpha'\)-completed "
        r"frameworks, of the form \(\sum_{k}\big[\Delta(X_{1k})-\Delta(X_{kn})\big]\,A(1\ldots k)\,A(k\ldots n)=0\)."
    )
    html = tex_to_html(tldr, markdown=True)
    assert "<b>quadratic sum rules</b>" in html
    assert "α'-completed" in html
    assert "∑<sub>k</sub>[Δ(X<sub>1k</sub>)-Δ(X<sub>kn</sub>)] A(1…k) A(k…n)=0" in html
    assert "**" not in html and "\\" not in html


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (r"Large-$N$ expansion in SU($N$) theories", "Large-N expansion in SU(N) theories"),
        (r"Preserving $\mathcal{W}_3^{\mathbb{Z}_2}$ Symmetry", "Preserving 𝒲<sub>3</sub><sup>ℤ<sub>2</sub></sup> Symmetry"),
        (r"$K_1^*(1870)$ and $3S$-$2D$ mixing", "K<sub>1</sub><sup>*</sup>(1870) and 3S-2D mixing"),
        (r"Display \[x^2\] and $$y_{\mathrm{eff}}$$", "Display x<sup>2</sup> and y<sub>eff</sub>"),
        (r"Integrability of the Schr\"odinger Equation", "Integrability of the Schrödinger Equation"),
    ],
)
def test_title_math_and_accents(title, expected):
    assert tex_to_html(title) == expected


def test_markdown_is_only_rendered_when_asked():
    assert tex_to_html("**bold** and *italic*") == "**bold** and *italic*"
    assert tex_to_html("**bold** and *italic*", markdown=True) == "<b>bold</b> and <i>italic</i>"


def test_bold_may_span_math():
    assert tex_to_html(r"**the \(\alpha_s\) term**", markdown=True) == "<b>the α<sub>s</sub> term</b>"


def test_html_is_escaped_and_plain_asterisks_survive():
    html = tex_to_html(r"a < b & c, 2*3*4 and $x < y$", markdown=True)
    assert html == "a &lt; b &amp; c, 2*3*4 and x &lt; y"


def test_malformed_latex_does_not_raise():
    assert tex_to_html(r"price \$10 and unbalanced $x and {brace") == "price $10 and unbalanced $x and brace"


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_text(empty):
    assert tex_to_html(empty) == ""
