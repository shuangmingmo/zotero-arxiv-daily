from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

AFFILIATION_PROMPT_TOKENS = 2000
# The author block of a LaTeX source: the first \author, \affiliation, ... call (a name followed by
# an argument, so macro definitions such as \newcommand{\address} are skipped), else \begin{document}.
_TEX_AUTHOR_BLOCK = re.compile(r'\\(?:author|affiliation|affil|institute|address)\s*[\[{]')
_TEX_DOCUMENT_START = re.compile(r'\\begin\{document\}')

AFFILIATION_SYSTEM_PROMPT = (
    "You are an assistant who perfectly extracts affiliations of authors from a paper. "
    "Return a JSON list of the authors' institutions in author order, like "
    "[\"Charles University\", \"University of Amsterdam\"]. "
    "Give only the name of the top-level institution: leave out departments, groups, and institutes "
    "or centres that belong to a university, and leave out street addresses, postcodes, cities and countries. "
    "An institution that is not part of a university keeps its own name. "
    "Examples: "
    "'Institute for Particle and Nuclear Physics, Charles University, Prague, Czech Republic' -> \"Charles University\"; "
    "'Institute for Theoretical Physics, University of Amsterdam, Science Park 904, 1098 XH Amsterdam, The Netherlands' -> \"University of Amsterdam\"; "
    "'Theoretical Physics Department, CERN, 1211 Geneva 23, Switzerland' -> \"CERN\"; "
    "'Perimeter Institute for Theoretical Physics, Waterloo, Ontario, Canada' -> \"Perimeter Institute for Theoretical Physics\". "
    "Write names as plain text without LaTeX, and list each institution once. "
    "If there is no affiliation, return an empty list []. Return only the list, with no other text."
)


def _affiliation_context(full_text: str) -> str:
    """Start the text sent for affiliation extraction at the author block.

    For LaTeX sources the preamble (packages, macros, TikZ styles) can run to thousands of
    tokens, which pushed the authors and affiliations out of the truncated prompt.
    """
    match = _TEX_AUTHOR_BLOCK.search(full_text) or _TEX_DOCUMENT_START.search(full_text)
    return full_text[match.start():] if match else full_text


def _request_llm(openai_client: OpenAI, llm_params: dict, messages: list[dict]) -> str:
    api_mode = llm_params.get("api_mode", "chat_completion")
    generation_kwargs = dict(llm_params.get("generation_kwargs", {}))

    if api_mode == "chat_completion":
        response = openai_client.chat.completions.create(
            messages=messages,
            **generation_kwargs,
        )
        return response.choices[0].message.content

    if api_mode == "response":
        max_tokens = generation_kwargs.pop("max_tokens", None)
        if max_tokens is not None and "max_output_tokens" not in generation_kwargs:
            generation_kwargs["max_output_tokens"] = max_tokens
        response = openai_client.responses.create(
            input=messages,
            **generation_kwargs,
        )
        return response.output_text

    raise ValueError(
        f"Unsupported llm.api_mode: {api_mode}. "
        "Expected 'chat_completion' or 'response'."
    )


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        tldr = _request_llm(
            openai_client,
            llm_params,
            [
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return tldr
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors as a JSON list in author order. If there is no affiliation found, return an empty list '[]':\n\n{_affiliation_context(self.full_text)}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:AFFILIATION_PROMPT_TOKENS]
            prompt = enc.decode(prompt_tokens)
            affiliations = _request_llm(
                openai_client,
                llm_params,
                [
                    {"role": "system", "content": AFFILIATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            # Drop blanks and repeats but keep author order (a set would shuffle it).
            affiliations = [str(a).strip() for a in affiliations]
            affiliations = list(dict.fromkeys(a for a in affiliations if a))

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]