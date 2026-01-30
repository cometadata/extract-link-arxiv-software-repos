"""Uses evamxb/dev-author-em-clf (98.56% accuracy) to match paper authors to GitHub contributors."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import Pipeline, pipeline

logger = logging.getLogger(__name__)

AUTHOR_MATCHING_MODEL = "evamxb/dev-author-em-clf"

# CRITICAL: This template format must be preserved exactly.
# The model was trained on this specific XML structure.
MODEL_INPUT_TEMPLATE = """
<developer-details>
    <username>{dev_username}</username>
    <name>{dev_name}</name>
    <email>{dev_email}</email>
</developer-details>

---

<author-details>
    <name>{author_name}</name>
</author-details>
""".strip()


@dataclass
class ContributorInfo:
    login: str
    name: Optional[str] = None
    email: Optional[str] = None


@dataclass
class AuthorMatchDetail:
    contributor_login: str
    author_name: str
    confidence: float


@dataclass
class AuthorMatchResult:
    matched: bool
    matches: List[AuthorMatchDetail] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None


def find_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_author_matching_model(device: Optional[str] = None, batch_size: int = 32) -> Pipeline:
    if device is None:
        device = find_device()

    logger.info(f"Loading author matching model on {device} with batch_size={batch_size}")

    return pipeline(
        "text-classification",
        model=AUTHOR_MATCHING_MODEL,
        device=device,
        batch_size=batch_size,
    )


def format_matching_input(
    username: str,
    name: Optional[str],
    email: Optional[str],
    author_name: str,
) -> str:
    return MODEL_INPUT_TEMPLATE.format(
        dev_username=username,
        dev_name=name,
        dev_email=email,
        author_name=author_name,
    )


DEFAULT_MAX_CONTRIBUTORS = 20


def match_authors_to_contributors(
    contributors: List[ContributorInfo],
    paper_authors: List[str],
    loaded_model: Optional[Pipeline] = None,
    device: Optional[str] = None,
    max_contributors: int = DEFAULT_MAX_CONTRIBUTORS,
    early_exit: bool = True,
) -> AuthorMatchResult:
    if not contributors:
        return AuthorMatchResult(
            matched=False,
            skipped=True,
            skip_reason="no_contributors",
        )

    if not paper_authors:
        return AuthorMatchResult(
            matched=False,
            skipped=True,
            skip_reason="no_authors",
        )

    if loaded_model is None:
        model = load_author_matching_model(device)
    else:
        model = loaded_model

    limited_contributors = contributors[:max_contributors]

    if len(contributors) > max_contributors:
        logger.debug(f"Limiting contributors from {len(contributors)} to {max_contributors}")

    inputs = []
    pairs = []

    for contributor in limited_contributors:
        for author in paper_authors:
            text = format_matching_input(
                username=contributor.login,
                name=contributor.name,
                email=contributor.email,
                author_name=author,
            )
            inputs.append(text)
            pairs.append((contributor, author))

    logger.debug(f"Running author matching on {len(inputs)} pairs")

    if early_exit and len(inputs) > 0:
        matches = []
        batch_size = 32

        for batch_start in range(0, len(inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(inputs))
            batch_inputs = inputs[batch_start:batch_end]
            batch_pairs = pairs[batch_start:batch_end]

            outputs = model(batch_inputs)

            for (contributor, author), output in zip(batch_pairs, outputs):
                if output["label"] == "match":
                    matches.append(AuthorMatchDetail(
                        contributor_login=contributor.login,
                        author_name=author,
                        confidence=output["score"],
                    ))
                    if early_exit:
                        logger.debug(f"Early exit: found match {contributor.login} <-> {author}")
                        return AuthorMatchResult(
                            matched=True,
                            matches=matches,
                        )

        return AuthorMatchResult(
            matched=len(matches) > 0,
            matches=matches,
        )

    outputs = model(inputs)

    matches = []
    for (contributor, author), output in zip(pairs, outputs):
        if output["label"] == "match":
            matches.append(AuthorMatchDetail(
                contributor_login=contributor.login,
                author_name=author,
                confidence=output["score"],
            ))

    return AuthorMatchResult(
        matched=len(matches) > 0,
        matches=matches,
    )


def batch_match_authors(
    records_data: List[Dict[str, Any]],
    model: Optional[Pipeline] = None,
    max_contributors: int = DEFAULT_MAX_CONTRIBUTORS,
    model_batch_size: int = 256,
) -> Dict[tuple, AuthorMatchResult]:
    """Batch matching across records. Each record needs 'key', 'contributors', and 'authors'."""
    if not records_data:
        return {}

    results = {}
    records_with_contributors = []
    for record in records_data:
        key = record["key"]
        contributors = record["contributors"]
        authors = record["authors"]

        if not contributors:
            results[key] = AuthorMatchResult(
                matched=False,
                skipped=True,
                skip_reason="no_contributors",
            )
        elif not authors:
            results[key] = AuthorMatchResult(
                matched=False,
                skipped=True,
                skip_reason="no_authors",
            )
        else:
            records_with_contributors.append(record)

    if not records_with_contributors:
        return results

    if model is None:
        for record in records_with_contributors:
            results[record["key"]] = AuthorMatchResult(
                matched=False,
                skipped=True,
                skip_reason="no_model",
            )
        return results

    all_inputs = []
    input_index = []

    for record in records_with_contributors:
        key = record["key"]
        contributors = record["contributors"][:max_contributors]
        authors = record["authors"]

        for contributor in contributors:
            for author in authors:
                text = format_matching_input(
                    username=contributor.login,
                    name=contributor.name,
                    email=contributor.email,
                    author_name=author,
                )
                all_inputs.append(text)
                input_index.append((key, contributor.login, author))

    outputs = model(all_inputs, batch_size=model_batch_size)

    matches_by_key = defaultdict(list)
    for (key, login, author), output in zip(input_index, outputs):
        if output["label"] == "match":
            matches_by_key[key].append(
                AuthorMatchDetail(
                    contributor_login=login,
                    author_name=author,
                    confidence=output["score"],
                )
            )

    for record in records_with_contributors:
        key = record["key"]
        matches = matches_by_key.get(key, [])
        results[key] = AuthorMatchResult(
            matched=len(matches) > 0,
            matches=matches,
        )

    return results
