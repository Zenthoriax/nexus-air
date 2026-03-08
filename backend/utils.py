import re
from typing import List


def parse_wikilinks(content: str) -> List[str]:
    """
    Parses wikilinks from a markdown string.

    Handles:
      [[Title]]            → 'Title'
      [[Title|Alias]]      → 'Title'  (alias stripped)
      Multiple links in one document — all returned.
      Duplicate link targets — deduplicated.

    Examples:
        >>> parse_wikilinks("See [[Python]] and [[Rust|Rust Lang]]")
        ['Python', 'Rust']
    """
    if not content:
        return []

    # Capture the target portion only; discard anything after a | (alias)
    links = re.findall(r'\[\[([^\|\]]+)(?:\|[^\]]+)?\]\]', content)
    return list(dict.fromkeys(link.strip() for link in links if link.strip()))
