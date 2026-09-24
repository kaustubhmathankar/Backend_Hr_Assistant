from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# RUNTIME RAG CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RAGRuntimeConfig:
    """Runtime RAG settings loaded from PostgreSQL."""

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    relevance_threshold: float = 0.35



INDEX_SCHEMA_VERSION = 1


def build_index_signature(
    config: Optional[RAGRuntimeConfig] = None,
) -> str:
    """Build a deterministic signature for index-affecting settings."""

    config = config or default_rag_config()

    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "chunk_size": int(config.chunk_size),
        "chunk_overlap": int(config.chunk_overlap),
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:16]


def is_index_current(
    document: Document,
    config: Optional[RAGRuntimeConfig] = None,
) -> bool:
    """Return True when a chunk matches the current index settings."""

    config = config or default_rag_config()
    metadata = document.metadata
    stored_signature = metadata.get("index_signature")

    # Existing chunks created before index signatures were introduced
    # used the original defaults. Keep them valid only while the current
    # chunking configuration still matches those defaults.
    if not stored_signature:
        return (
            config.chunk_size == 1000
            and config.chunk_overlap == 150
        )

    return str(stored_signature) == build_index_signature(config)


def default_rag_config() -> RAGRuntimeConfig:
    """Return the application defaults for RAG settings."""

    return RAGRuntimeConfig(
        chunk_size=1000,
        chunk_overlap=150,
        retrieval_top_k=5,
        relevance_threshold=0.35,
    )


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    encode_kwargs={
        "normalize_embeddings": True,
    },
)


# ============================================================
# CHROMA
# ============================================================

chroma_path = Path(
    settings.CHROMA_PERSIST_DIRECTORY
)

chroma_path.mkdir(
    parents=True,
    exist_ok=True,
)

vector_store = Chroma(
    collection_name=settings.CHROMA_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(
        chroma_path
    ),
)


# ============================================================
# TEXT SPLITTER
# ============================================================

def build_text_splitter(
    config: Optional[RAGRuntimeConfig] = None,
) -> RecursiveCharacterTextSplitter:
    """Build a text splitter using the supplied runtime settings."""

    config = config or default_rag_config()

    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if config.chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if config.chunk_overlap >= config.chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(
    documents: list[Document],
    config: Optional[RAGRuntimeConfig] = None,
) -> list[Document]:
    if not documents:
        return []

    splitter = build_text_splitter(config)
    return splitter.split_documents(documents)


# ============================================================
# NORMALIZE METADATA
# ============================================================

def normalize_document_metadata(
    document: Document,
) -> Document:
    """Normalize identifiers stored in Chroma metadata."""

    metadata = dict(
        document.metadata
    )

    if "user_id" in metadata:
        metadata["user_id"] = str(
            metadata["user_id"]
        )

    if "file_id" in metadata:
        metadata["file_id"] = str(
            metadata["file_id"]
        )

    document.metadata = metadata
    return document


# ============================================================
# ADD DOCUMENTS TO CHROMA
# ============================================================

def add_documents_to_chroma(
    documents: list[Document],
    config: Optional[RAGRuntimeConfig] = None,
) -> int:
    if not documents:
        return 0

    config = config or default_rag_config()
    index_signature = build_index_signature(config)

    normalized_documents: list[Document] = []
    ids: list[str] = []

    for index, document in enumerate(
        documents
    ):
        document = normalize_document_metadata(
            document
        )

        document.metadata["index_schema_version"] = INDEX_SCHEMA_VERSION
        document.metadata["index_signature"] = index_signature
        document.metadata["index_chunk_size"] = int(config.chunk_size)
        document.metadata["index_chunk_overlap"] = int(config.chunk_overlap)

        user_id = document.metadata.get(
            "user_id",
            "unknown",
        )
        file_id = document.metadata.get(
            "file_id",
            "unknown",
        )

        chunk_id = (
            f"{user_id}-"
            f"{file_id}-"
            f"{index}"
        )

        ids.append(chunk_id)
        normalized_documents.append(document)

    vector_store.add_documents(
        documents=normalized_documents,
        ids=ids,
    )

    return len(normalized_documents)


# ============================================================
# DELETE FILE FROM CHROMA
# ============================================================

def delete_file_from_chroma(
    file_id: int,
) -> int:
    normalized_file_id = str(file_id)

    collection = getattr(
        vector_store,
        "_collection",
        None,
    )

    if collection is None:
        raise RuntimeError(
            "Chroma collection is not available."
        )

    existing = collection.get(
        where={
            "file_id": normalized_file_id,
        },
        include=[],
    )

    existing_ids = existing.get(
        "ids",
        [],
    )

    chunk_count = len(existing_ids)

    if not existing_ids:
        return 0

    collection.delete(
        ids=existing_ids
    )

    print(
        "[RAG] Deleted "
        f"{chunk_count} Chroma chunks "
        f"for file_id={file_id}"
    )

    return chunk_count


# ============================================================
# INGEST DOCUMENTS
# ============================================================

def ingest_documents(
    documents: list[Document],
    config: Optional[RAGRuntimeConfig] = None,
) -> dict:
    if not documents:
        raise ValueError(
            "No readable documents were produced."
        )

    chunks = split_documents(
        documents,
        config=config,
    )

    if not chunks:
        raise ValueError(
            "No usable chunks were created."
        )

    chunk_count = add_documents_to_chroma(
        chunks,
        config=config,
    )

    extracted_characters = sum(
        len(document.page_content)
        for document in documents
    )

    return {
        "documents": len(documents),
        "chunks": chunk_count,
        "extracted_characters": extracted_characters,
    }


# ============================================================
# INGEST FILE
# ============================================================

def ingest_file(
    filename: str,
    file_bytes: bytes,
    user_id: int,
    file_id: int,
    rag_config: Optional[RAGRuntimeConfig] = None,
    max_zip_files: Optional[int] = None,
    max_zip_uncompressed_size: Optional[int] = None,
) -> dict:
    """Extract, chunk, and index an uploaded file."""

    from app.file_processor import extract_documents

    if max_zip_files is None:
        max_zip_files = settings.MAX_ZIP_FILES

    if max_zip_uncompressed_size is None:
        max_zip_uncompressed_size = (
            settings.MAX_ZIP_UNCOMPRESSED_SIZE_MB
            * 1024
            * 1024
        )

    config = rag_config or default_rag_config()

    print(
        "\n[RAG INGEST CONFIG] "
        f"chunk_size={config.chunk_size} | "
        f"chunk_overlap={config.chunk_overlap}"
    )

    documents = extract_documents(
        filename=filename,
        file_bytes=file_bytes,
        user_id=user_id,
        file_id=file_id,
        max_zip_files=max_zip_files,
        max_zip_uncompressed_size=max_zip_uncompressed_size,
    )

    result = ingest_documents(
        documents,
        config=config,
    )

    # Keep the ingestion response contract compatible with the
    # upload router. The router uses these values when building
    # the API response, while ingest_documents() historically
    # returned only document/chunk statistics.
    result["chunk_size"] = int(config.chunk_size)
    result["chunk_overlap"] = int(config.chunk_overlap)

    return result


# ============================================================
# VERIFY OWNERSHIP + SELECTED FILE
# ============================================================

def _belongs_to_selected_file(
    document: Document,
    user_id: int,
    file_ids: set[int],
) -> bool:
    """Verify tenant ownership and explicit file selection."""

    metadata = document.metadata

    stored_user_id = metadata.get("user_id")
    stored_file_id = metadata.get("file_id")

    if stored_user_id is None or stored_file_id is None:
        return False

    try:
        normalized_user_id = int(stored_user_id)
    except (TypeError, ValueError):
        return False

    try:
        normalized_file_id = int(stored_file_id)
    except (TypeError, ValueError):
        return False

    return (
        normalized_user_id == user_id
        and normalized_file_id in file_ids
    )


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def _normalize_distance(
    distance: float,
) -> float:
    """
    Convert Chroma distance into a bounded similarity-like score.

    This is a calibration heuristic, not a native Chroma relevance
    score. Smaller distance produces a higher score in the range
    (0, 1].
    """

    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0

    if value < 0:
        value = 0.0

    return 1.0 / (1.0 + value)


# ============================================================
# DIRECT CHROMA SEARCH
# ============================================================

def _filtered_chroma_search(
    query: str,
    file_id: int,
    top_k: int,
) -> list[tuple[Document, float]]:
    """
    Filter only by file_id.

    Chroma is currently rejecting the previous two-key metadata
    filter form. User ownership is still verified in Python before
    any result is returned.
    """

    normalized_file_id = str(file_id)

    try:
        raw_results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter={
                "file_id": normalized_file_id,
            },
        )
    except Exception as exc:
        print(
            "[RAG] File metadata search failed: "
            f"{exc}"
        )
        return []

    results: list[tuple[Document, float]] = []

    for document, distance in raw_results or []:
        results.append(
            (
                document,
                _normalize_distance(distance),
            )
        )

    return results


# ============================================================
# SAFE GLOBAL FALLBACK
# ============================================================

def _fallback_global_search(
    query: str,
    user_id: int,
    file_ids: set[int],
    top_k: int,
) -> list[tuple[Document, float]]:
    """
    Wider retrieval followed by explicit user/file verification.
    """

    candidate_k = max(
        top_k * 10,
        50,
    )

    try:
        raw_results = vector_store.similarity_search_with_score(
            query=query,
            k=candidate_k,
        )
    except Exception as exc:
        print(
            "[RAG] Global fallback search failed: "
            f"{exc}"
        )
        return []

    filtered: list[tuple[Document, float]] = []

    for document, distance in raw_results or []:
        if not _belongs_to_selected_file(
            document=document,
            user_id=user_id,
            file_ids=file_ids,
        ):
            continue

        filtered.append(
            (
                document,
                _normalize_distance(distance),
            )
        )

    return filtered


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query: str,
    user_id: int,
    file_ids: list[int],
    config: Optional[RAGRuntimeConfig] = None,
) -> list[tuple[Document, float]]:
    """
    Retrieve ONLY from explicitly selected files.

    Security boundary:
        authenticated user_id
                +
        explicitly selected file_ids

    The runtime retrieval settings are supplied through config.
    """

    config = config or default_rag_config()

    if config.retrieval_top_k <= 0:
        raise ValueError(
            "retrieval_top_k must be greater than 0."
        )

    if not 0.0 <= config.relevance_threshold <= 1.0:
        raise ValueError(
            "relevance_threshold must be between 0 and 1."
        )

    query = query.strip()

    if not query or not file_ids:
        return []

    unique_file_ids = set(file_ids)

    print(
        "\n========== RAG RETRIEVAL =========="
    )
    print(
        f"User ID: {user_id}"
    )
    print(
        f"Selected File IDs: {sorted(unique_file_ids)}"
    )
    print(
        f"Query: {query}"
    )
    print(
        "[RAG CONFIG] "
        f"top_k={config.retrieval_top_k} | "
        f"threshold={config.relevance_threshold} | "
        f"chunk_size={config.chunk_size} | "
        f"chunk_overlap={config.chunk_overlap}"
    )

    all_results: list[tuple[Document, float]] = []

    # --------------------------------------------------------
    # Direct retrieval per selected file
    # --------------------------------------------------------

    for file_id in sorted(unique_file_ids):
        file_results = _filtered_chroma_search(
            query=query,
            file_id=file_id,
            top_k=config.retrieval_top_k,
        )

        print(
            f"Direct retrieval file_id={file_id}: "
            f"{len(file_results)} chunks"
        )

        all_results.extend(file_results)

    # --------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------

    if not all_results:
        print(
            "[RAG] Direct metadata retrieval returned no chunks."
        )

        fallback_results = _fallback_global_search(
            query=query,
            user_id=user_id,
            file_ids=unique_file_ids,
            top_k=config.retrieval_top_k,
        )

        print(
            "[RAG] Safe fallback retrieved "
            f"{len(fallback_results)} chunks."
        )

        all_results.extend(fallback_results)

    # --------------------------------------------------------
    # Final security verification
    # --------------------------------------------------------

    verified_results: list[tuple[Document, float]] = []
    stale_count = 0

    for document, score in all_results:
        if not _belongs_to_selected_file(
            document=document,
            user_id=user_id,
            file_ids=unique_file_ids,
        ):
            continue

        if not is_index_current(
            document=document,
            config=config,
        ):
            stale_count += 1
            continue

        verified_results.append(
            (
                document,
                score,
            )
        )

    if not verified_results:
        if stale_count > 0:
            print(
                "[RAG] Selected file chunks are stale for the "
                "current index configuration. Re-index required."
            )

        print(
            "[RAG] No chunks passed "
            "user/file ownership verification and index checks."
        )
        print(
            "===================================\n"
        )
        return []

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    unique_results: dict[
        tuple,
        tuple[Document, float],
    ] = {}

    for document, score in verified_results:
        metadata = document.metadata

        identity = (
            str(metadata.get("user_id")),
            str(metadata.get("file_id")),
            metadata.get("source"),
            metadata.get("page"),
            metadata.get("sheet"),
            metadata.get("slide"),
            document.page_content,
        )

        existing = unique_results.get(identity)

        if existing is None or score > existing[1]:
            unique_results[identity] = (
                document,
                score,
            )

    final_results = list(
        unique_results.values()
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    final_results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    # --------------------------------------------------------
    # Apply runtime relevance threshold
    # --------------------------------------------------------

    threshold_results = [
        item
        for item in final_results
        if item[1] >= config.relevance_threshold
    ]

    # --------------------------------------------------------
    # Apply runtime top-k
    # --------------------------------------------------------

    final_results = threshold_results[
        :config.retrieval_top_k
    ]

    print(
        f"[RAG] Candidates after verification: "
        f"{len(verified_results)}"
    )
    print(
        f"[RAG] Stale-index candidates excluded: "
        f"{stale_count}"
    )
    print(
        f"[RAG] Candidates after threshold: "
        f"{len(threshold_results)}"
    )
    print(
        f"[RAG] Final chunks: "
        f"{len(final_results)}"
    )

    for document, score in final_results:
        print(
            "  "
            f"file_id={document.metadata.get('file_id')} "
            f"user_id={document.metadata.get('user_id')} "
            f"source={document.metadata.get('source')} "
            f"score={score:.4f}"
        )

    print(
        "===================================\n"
    )

    return final_results


# ============================================================
# FILE INDEX STATUS
# ============================================================

def get_file_index_status(
    file_id: int,
    config: Optional[RAGRuntimeConfig] = None,
) -> dict:
    """Inspect whether a file has current Chroma index chunks."""

    config = config or default_rag_config()
    collection = getattr(vector_store, "_collection", None)

    if collection is None:
        raise RuntimeError(
            "Chroma collection is not available."
        )

    result = collection.get(
        where={
            "file_id": str(file_id),
        },
        include=["metadatas"],
    )

    ids = result.get("ids", []) or []
    metadatas = result.get("metadatas", []) or []
    expected_signature = build_index_signature(config)

    current = 0
    stale = 0

    for metadata in metadatas:
        metadata = metadata or {}
        stored_signature = metadata.get("index_signature")

        if (
            not stored_signature
            and config.chunk_size == 1000
            and config.chunk_overlap == 150
        ):
            current += 1
        elif str(stored_signature) == expected_signature:
            current += 1
        else:
            stale += 1

    if not ids:
        status = "missing"
    elif current == len(ids):
        status = "current"
    else:
        status = "stale"

    return {
        "file_id": int(file_id),
        "status": status,
        "chunk_count": len(ids),
        "current_chunks": current,
        "stale_chunks": stale,
        "expected_index_signature": expected_signature,
        "expected_chunk_size": int(config.chunk_size),
        "expected_chunk_overlap": int(config.chunk_overlap),
    }


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results: list[tuple[Document, float]],
) -> str:
    context_parts: list[str] = []

    for index, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "unknown",
        )

        location = ""

        if "page" in document.metadata:
            location = (
                f" - Page "
                f"{document.metadata['page']}"
            )
        elif "sheet" in document.metadata:
            location = (
                f" - Sheet "
                f"{document.metadata['sheet']}"
            )
        elif "slide" in document.metadata:
            location = (
                f" - Slide "
                f"{document.metadata['slide']}"
            )

        context_parts.append(
            f"[Document {index}]\n"
            f"Source: {source}{location}\n"
            f"Relevance: {score:.4f}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# SOURCE NAMES
# ============================================================

def get_source_names(
    results: list[tuple[Document, float]],
) -> list[str]:
    sources: list[str] = []

    for document, _score in results:
        source = document.metadata.get("source")

        if source and source not in sources:
            sources.append(source)

    return sources
