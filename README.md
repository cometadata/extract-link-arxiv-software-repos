# extract-software-repos

Extract and validate software repository URLs from scientific papers, with ML-powered promotion to identify official implementations.

## Installation

```bash
pip install -e .
```

Or with uv:
```bash
uv pip install -e .
```

## Usage

### Extract URLs from Papers

```bash
# From Parquet full-text (with text healing)
extract-software-repos extract papers.parquet -o enrichments.jsonl --heal-fulltext

# From DataCite abstracts
extract-software-repos extract records.jsonl.gz --from-datacite-abstract -o enrichments.jsonl
```

### Validate and Promote

Validate URLs exist and promote repos to `isSupplementedBy` when heuristics indicate official implementation:

```bash
export GITHUB_TOKEN=ghp_your_token_here
extract-software-repos validate enrichments.jsonl \
  --promote \
  --records records.jsonl.gz \
  --github-cache github_cache.jsonl \
  -o enrichments_final.jsonl
```

Promotion uses three signals (requires 2+ by default):
- arXiv ID detection in repo README/description
- Name similarity between repo and paper title
- Author matching via ML model

### Merge with DataCite Records

Output is compatible with [datacite-enrichment](https://github.com/cometadata/datacite-enrichment):

```bash
datacite-enrich merge records.jsonl.gz enrichments_final.jsonl -o merged.jsonl
```

## Acknowledgments

The author matching model and inference code are adapted from work by Eva Maxfield Brown:

**Code Repository:**
Brown, E. M. (2025). sci-soft-models [Computer software]. GitHub. https://github.com/evamaxfield/sci-soft-models

**Model:**
Brown, E. M. (2025). evamaxfield/sci-soft-models: Author-Developer Account Matching (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17401862

**Paper:**
Brown, E. M., Slaughter, I., & Weber, N. (2025). Code Contribution and Credit in Science. arXiv. https://doi.org/10.48550/arXiv.2510.16242

## Development

```bash
uv pip install -e ".[dev]"
pytest tests/ -v
```
