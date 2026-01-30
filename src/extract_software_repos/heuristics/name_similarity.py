import string
from dataclasses import dataclass
from typing import Optional, Set

from Levenshtein import ratio as levenshtein_ratio


@dataclass
class NameSimilarityResult:
    matched: bool
    score: float = 0.0
    containment_score: float = 0.0
    token_overlap_score: float = 0.0
    fuzzy_score: float = 0.0
    skipped: bool = False
    skip_reason: Optional[str] = None


STOPWORDS = frozenset([
    "a", "an", "the", "for", "of", "on", "in", "with", "to", "and", "or",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "it", "its",
    "by", "from", "at", "as",
])


def normalize_text(text: str) -> Set[str]:
    """Lowercase, replace hyphens/underscores, remove punctuation and stopwords, return token set."""
    text = text.lower()
    text = text.replace("-", " ").replace("_", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = set(text.split())
    tokens = tokens - STOPWORDS

    return tokens


def compute_containment_score(repo_name: str, paper_title: str) -> float:
    """Max ratio of shared tokens to either set's size."""
    repo_tokens = normalize_text(repo_name)
    title_tokens = normalize_text(paper_title)

    if not repo_tokens or not title_tokens:
        return 0.0

    repo_in_title = len(repo_tokens & title_tokens) / len(repo_tokens)
    title_in_repo = len(repo_tokens & title_tokens) / len(title_tokens)

    return max(repo_in_title, title_in_repo)


def compute_token_overlap_score(repo_name: str, paper_title: str) -> float:
    """Jaccard similarity of token sets."""
    repo_tokens = normalize_text(repo_name)
    title_tokens = normalize_text(paper_title)

    if not repo_tokens or not title_tokens:
        return 0.0

    intersection = len(repo_tokens & title_tokens)
    union = len(repo_tokens | title_tokens)

    return intersection / union if union > 0 else 0.0


def compute_fuzzy_score(repo_name: str, paper_title: str) -> float:
    """Levenshtein ratio between sorted normalized token strings."""
    repo_normalized = " ".join(sorted(normalize_text(repo_name)))
    title_normalized = " ".join(sorted(normalize_text(paper_title)))

    if not repo_normalized or not title_normalized:
        return 0.0

    return levenshtein_ratio(repo_normalized, title_normalized)


def compute_name_similarity(
    repo_name: Optional[str],
    paper_title: Optional[str],
    threshold: float = 0.45,
    containment_weight: float = 0.4,
    overlap_weight: float = 0.4,
    fuzzy_weight: float = 0.2,
) -> NameSimilarityResult:
    if not repo_name:
        return NameSimilarityResult(
            matched=False,
            skipped=True,
            skip_reason="no_repo_name",
        )

    if not paper_title:
        return NameSimilarityResult(
            matched=False,
            skipped=True,
            skip_reason="no_paper_title",
        )

    containment = compute_containment_score(repo_name, paper_title)
    overlap = compute_token_overlap_score(repo_name, paper_title)
    fuzzy = compute_fuzzy_score(repo_name, paper_title)

    final_score = (
        containment_weight * containment +
        overlap_weight * overlap +
        fuzzy_weight * fuzzy
    )

    return NameSimilarityResult(
        matched=final_score >= threshold,
        score=final_score,
        containment_score=containment,
        token_overlap_score=overlap,
        fuzzy_score=fuzzy,
    )
