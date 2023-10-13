import regex as re

_ONLY_WHITE_SPACE = re.compile(r"^\s*$")
_MULTI_WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Replace all duplicate whitespaces with a single whitespace"""

    return _MULTI_WHITESPACE_REGEX.sub(" ", text).strip()


def is_only_whitespace(s: str) -> bool:
    return bool(_ONLY_WHITE_SPACE.match(s))
