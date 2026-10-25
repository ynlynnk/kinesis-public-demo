from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"

FRONTMATTER_PATTERN = re.compile(
    r"\A---\s*\r?\n.*?\r?\n---\s*(?:\r?\n|$)",
    re.DOTALL,
)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
URL_PATTERN = re.compile(r"https?://[^\s)>]+")


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    document_title: str
    section_title: str
    text: str
    links: tuple[str, ...]


def remove_frontmatter(text: str) -> str:
    return FRONTMATTER_PATTERN.sub("", text, count=1).strip()


def read_document_chunks(path: Path) -> list[KnowledgeChunk]:
    content = remove_frontmatter(
        path.read_text(encoding="utf-8-sig")
    )

    document_title = path.stem.replace("_", " ").title()
    section_title = document_title
    section_lines: list[str] = []
    chunks: list[KnowledgeChunk] = []
    document_links = tuple(dict.fromkeys(URL_PATTERN.findall(content)))

    def save_section() -> None:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            return

        chunks.append(
            KnowledgeChunk(
                source=path.relative_to(PROJECT_ROOT).as_posix(),
                document_title=document_title,
                section_title=section_title,
                text=section_text,
                links=document_links,
            )
        )

    for line in content.splitlines():
        heading = HEADING_PATTERN.match(line)

        if heading:
            save_section()
            section_lines = []
            heading_level = len(heading.group(1))
            heading_text = heading.group(2).strip()

            if heading_level == 1:
                document_title = heading_text

            section_title = heading_text
            continue

        section_lines.append(line)

    save_section()
    return chunks


def load_knowledge_chunks() -> list[KnowledgeChunk]:
    if not KNOWLEDGE_ROOT.exists():
        raise FileNotFoundError(
            f"Knowledge folder not found: {KNOWLEDGE_ROOT}"
        )

    chunks: list[KnowledgeChunk] = []

    for path in sorted(KNOWLEDGE_ROOT.rglob("*.md")):
        chunks.extend(read_document_chunks(path))

    if not chunks:
        raise RuntimeError(
            "No readable Markdown knowledge documents were found."
        )

    return chunks


def retrieve_knowledge(
    query: str,
    top_k: int = 4,
    minimum_score: float = 0.03,
) -> list[dict]:
    clean_query = query.strip()

    if not clean_query:
        raise ValueError("A retrieval query is required.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    chunks = load_knowledge_chunks()

    searchable_text = [
        (
            f"{chunk.document_title}\n"
            f"{chunk.section_title}\n"
            f"{chunk.text}"
        )
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(
        [clean_query, *searchable_text]
    )

    scores = cosine_similarity(
        matrix[0:1],
        matrix[1:],
    ).flatten()

    ranked_results = sorted(
        zip(chunks, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []

    for chunk, score in ranked_results:
        if float(score) < minimum_score:
            continue

        results.append(
            {
                "source": chunk.source,
                "document_title": chunk.document_title,
                "section_title": chunk.section_title,
                "relevance_score": round(float(score), 4),
                "text": chunk.text,
                "links": list(chunk.links),
            }
        )

        if len(results) == top_k:
            break

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the curated KINESIS knowledge base."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="muscle fatigue and movement analysis",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
    )
    arguments = parser.parse_args()

    results = retrieve_knowledge(
        query=arguments.query,
        top_k=arguments.top_k,
    )

    print(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
