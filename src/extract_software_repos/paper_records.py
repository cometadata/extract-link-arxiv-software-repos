"""Paper record extraction for different metadata formats."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)


@dataclass
class PaperInfo:
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    arxiv_id: Optional[str] = None


def normalize_doi(doi: str) -> str:
    """Lowercase and strip URL prefixes (https://doi.org/, etc.)."""
    doi = doi.lower()

    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ]
    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break

    return doi


def extract_paper_info_datacite(record: Dict[str, Any]) -> PaperInfo:
    """Extract from DataCite record with attributes.doi, titles, creators, alternateIdentifiers."""
    attrs = record.get("attributes", {})

    doi = attrs.get("doi") or record.get("id")

    titles = attrs.get("titles", [])
    title = titles[0]["title"] if titles else None

    authors = []
    for creator in attrs.get("creators", []):
        if creator.get("nameType") != "Personal":
            continue

        if creator.get("givenName") and creator.get("familyName"):
            name = f"{creator['givenName']} {creator['familyName']}"
        else:
            name = creator.get("name", "")
            if ", " in name:
                parts = name.split(", ", 1)
                name = f"{parts[1]} {parts[0]}"

        if name:
            authors.append(name)

    arxiv_id = None

    for alt_id in attrs.get("alternateIdentifiers", []):
        if alt_id.get("alternateIdentifierType") == "arXiv":
            arxiv_id = alt_id.get("alternateIdentifier")
            break

    if not arxiv_id:
        for ident in attrs.get("identifiers", []):
            if ident.get("identifierType") == "arXiv":
                arxiv_id = ident.get("identifier")
                break

    return PaperInfo(
        doi=doi,
        title=title,
        authors=authors,
        arxiv_id=arxiv_id,
    )


RECORD_EXTRACTORS = {
    "datacite": extract_paper_info_datacite,
}


def extract_paper_info(record: Dict[str, Any], record_type: str = "datacite") -> PaperInfo:
    """Raises ValueError if record_type is unknown."""
    extractor = RECORD_EXTRACTORS.get(record_type)
    if extractor is None:
        raise ValueError(f"Unknown record type: {record_type}. Known types: {list(RECORD_EXTRACTORS.keys())}")

    return extractor(record)


def load_papers_for_dois(
    records_path: Path,
    dois_needed: Set[str],
    record_type: str = "datacite",
) -> List[PaperInfo]:
    """Use DuckDB to efficiently query large JSONL files for matching DOIs."""
    import duckdb

    if not dois_needed:
        return []

    logger.info(f"Loading papers for {len(dois_needed):,} DOIs from {records_path}")

    dois_list = list(dois_needed)

    if record_type == "datacite":
        query = """
            SELECT *
            FROM read_json_auto(?, maximum_object_size=104857600, ignore_errors=true)
            WHERE LOWER(COALESCE(attributes.doi, id)) IN (SELECT UNNEST(?::VARCHAR[]))
        """
    else:
        raise ValueError(f"Unknown record type: {record_type}")

    conn = duckdb.connect(":memory:")

    try:
        result = conn.execute(query, [str(records_path), dois_list]).fetchall()
        columns = [desc[0] for desc in conn.description]
    finally:
        conn.close()

    logger.info(f"Found {len(result):,} matching paper records")

    extractor = RECORD_EXTRACTORS.get(record_type)
    if extractor is None:
        raise ValueError(f"Unknown record type: {record_type}")

    papers = []
    for row in result:
        record = dict(zip(columns, row))
        papers.append(extractor(record))

    return papers
