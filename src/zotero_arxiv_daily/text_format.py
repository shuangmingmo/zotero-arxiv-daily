"""Turn the LaTeX in arXiv metadata and the Markdown/LaTeX in LLM output into readable email text.

arXiv titles and author names are TeX ("Ji\\v{r}\\'i Novotn\\'y", "$\\mathbb{Z}_2$ symmetry"), and the
TL;DR model writes Markdown and LaTeX ("**sum rules**", "\\(\\alpha'\\)"). Email clients render neither,
so this module converts them to Unicode text and plain HTML.
"""

import html
import logging
import re

from loguru import logger
from pylatexenc.latex2text import LatexNodes2Text

# pylatexenc logs a warning through the standard logging module for every macro it does not know.
logging.getLogger("pylatexenc").setLevel(logging.ERROR)

_CONVERTER = LatexNodes2Text()

# Inline and display math; a dollar sign written as \$ does not open math.
_MATH = re.compile(r"(?<!\\)\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|(?<!\\)\$(.+?)(?<!\\)\$", re.DOTALL)
# A TeX command in running text, with the argument it applies to: \v{r}, \'i, \emph{x}, {\O}, \o{}.
# Only these are converted outside math, so characters such as % and & in prose stay as they are.
_TEXT_COMMAND = re.compile(
    r"\{\\[^{}]*\}"                           # {\O}, {\"o}, {\i}
    r"|\\[A-Za-z]+(?:\s*\{[^{}]*\})*"         # \v{r}, \emph{word}, \o{}, \ss
    r"|\\[^A-Za-z\s](?:\{[^{}]*\}|[A-Za-z])?"  # \'i, \"{o}, \%, \&
)
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.DOTALL)
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_PLACEHOLDER = "\x00{}\x00"
_PLACEHOLDER_RE = re.compile("\x00(\\d+)\x00")


def _latex_to_text(tex: str) -> str:
    try:
        return _CONVERTER.latex_to_text(tex)
    except Exception as exc:  # malformed input: show it as written rather than lose it
        logger.debug(f"Could not convert LaTeX {tex!r}: {exc}")
        return tex


def latex_to_unicode(text: str) -> str:
    """Replace the TeX commands in running text with Unicode: "Ji\\v{r}\\'i" -> "Jiří"."""
    if "\\" not in text and "{" not in text:
        return text
    text = _TEXT_COMMAND.sub(lambda m: _latex_to_text(m.group(0)), text)
    # Remaining braces only group letters, as in "{QCD}" or "c{\i}" after conversion.
    return text.replace("{", "").replace("}", "")


def _group_end(tex: str, start: int) -> int:
    """Index just past the brace group that opens at tex[start], or len(tex) if it never closes."""
    depth = 0
    for i in range(start, len(tex)):
        if tex[i] == "{":
            depth += 1
        elif tex[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(tex)


def _script_argument(tex: str, start: int) -> tuple[str, int] | None:
    """The argument of a sub/superscript that begins at tex[start]: {group}, \\macro, or one character."""
    i = start
    while i < len(tex) and tex[i] == " ":
        i += 1
    if i >= len(tex):
        return None
    if tex[i] == "{":
        end = _group_end(tex, i)
        return tex[i + 1 : end - 1], end
    if tex[i] == "\\":
        end = i + len(re.match(r"\\(?:[A-Za-z]+|.)", tex[i:]).group(0))
        # A command keeps its brace arguments: _\mathrm{eff} is one subscript.
        while tex[i + 1 : end].isalpha() and end < len(tex) and tex[end] == "{":
            end = _group_end(tex, end)
        return tex[i:end], end
    return tex[i], i + 1


def _math_to_html(tex: str) -> str:
    """Render LaTeX math as Unicode HTML: "\\sum_{k} X_{1k}^2" -> "∑<sub>k</sub> X<sub>1k</sub><sup>2</sup>".

    Sub- and superscripts are found at the top level only; brace groups are passed whole to the
    converter so that commands keep their arguments (a script inside one stays as "_x").
    """
    out: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            out.append(html.escape(_latex_to_text("".join(pending)), quote=False))
            pending.clear()

    i = 0
    while i < len(tex):
        ch = tex[i]
        if ch in "_^":
            argument = _script_argument(tex, i + 1)
            if argument is not None:
                flush()
                tag = "sub" if ch == "_" else "sup"
                out.append(f"<{tag}>{_math_to_html(argument[0])}</{tag}>")
                i = argument[1]
                continue
        if ch == "{":
            end = _group_end(tex, i)
            pending.append(tex[i:end])
            i = end
            continue
        if ch == "\\":
            macro = re.match(r"\\(?:[A-Za-z]+|.)", tex[i:]).group(0)
            pending.append(macro)
            i += len(macro)
            continue
        pending.append(ch)
        i += 1
    flush()
    return "".join(out)


def tex_to_html(text: str | None, markdown: bool = False) -> str:
    """HTML for text that mixes prose with LaTeX math, e.g. an arXiv title or an LLM summary.

    With markdown=True, **bold** and *italic* are rendered too. Everything else is HTML-escaped.
    """
    if not text:
        return ""
    maths: list[str] = []

    def stash(match: re.Match) -> str:
        tex = next(group for group in match.groups() if group is not None)
        maths.append(_math_to_html(tex))
        return _PLACEHOLDER.format(len(maths) - 1)

    text = _MATH.sub(stash, text)
    text = html.escape(latex_to_unicode(text), quote=False)
    if markdown:
        text = _BOLD.sub(r"<b>\1</b>", text)
        text = _ITALIC.sub(r"<i>\1</i>", text)
    return _PLACEHOLDER_RE.sub(lambda m: maths[int(m.group(1))], text)
