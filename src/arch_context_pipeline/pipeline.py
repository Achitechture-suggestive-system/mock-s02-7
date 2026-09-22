from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


PROFILE_VERSION = "ARCH-KB-PUML/1.1.0"
VOCABULARY_VERSION = "1.1.4"
MOCK_SCHEMA_VERSION = "0.1.0-mock"
LOCAL_LLM_SCHEMA_VERSION = "0.1.0-local-llm"

RETRIEVAL_FIELDS = (
    "domain_tags",
    "problem_statement",
    "actor_descriptions",
    "functional_requirements",
    "non_functional_requirements",
    "eligible_constraints",
    "domain_terms",
)

# These are configuration defaults, not learned or research-derived constants.
# The repository has no relevance-labelled data that would justify unequal
# production field weights, so the production default is neutral. The legacy
# index weights remain visible in retrieval metadata for audit only.
DEFAULT_FIELD_WEIGHTS = {field_name: 1.0 for field_name in RETRIEVAL_FIELDS}
DEFAULT_FIELD_B = {field_name: 0.75 for field_name in RETRIEVAL_FIELDS}

# Kept only for the explicit legacy TF-IDF/debug tokenizer. Production
# retrieval does not use expansion, accent folding, stopword deletion or
# stemming.
LEGACY_VIETNAMESE_EXPANSIONS = {
    "phòng khám": "clinic healthcare medical practice",
    "đặt lịch": "appointment booking scheduling",
    "lịch hẹn": "appointment booking scheduling",
    "bác sĩ": "doctor clinician medical provider",
    "bệnh nhân": "patient",
    "lễ tân": "receptionist administrative user",
    "hủy lịch": "appointment cancellation",
    "hủy cuộc hẹn": "appointment cancellation",
    "dữ liệu": "data",
    "cơ sở dữ liệu": "database relational database",
    "bộ nhớ đệm": "cache",
    "hạ tầng nội bộ": "self-hosted internal infrastructure",
    "không ra ngoài": "no outbound external traffic",
    "đồng thời": "concurrent concurrent requests",
    "yêu cầu": "requirement request",
}

LEGACY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "shall", "that", "the", "their",
    "this", "to", "with", "will", "need", "needs", "system", "platform", "user",
    "shall", "must", "should", "cần", "có", "và", "là", "cho", "của", "trên",
    "một", "các", "được", "thì", "khi", "này", "từ", "với",
}

# These terms remain in the BM25F score and in `matched_terms`. They are used
# only by the retrieval audit so a generic overlap is not mistaken for strong
# domain support. The audit is a conservative heuristic, not a relevance
# model or an OOD classifier.
RETRIEVAL_AUDIT_GENERIC_TERMS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "with", "will", "shall", "must", "should", "only", "through",
    "use", "uses", "using", "system", "service", "services", "application",
    "applications", "platform", "software", "data", "information", "user", "users",
    "client", "clients", "request", "requests", "interface", "interfaces", "access",
    "support", "supports", "provide", "provides", "expose", "exposes", "external",
    "internal", "environment", "environments",
}
RETRIEVAL_AUDIT_MIN_KNOWN_CONTENT_TERMS = 2
RETRIEVAL_AUDIT_MIN_TOP_CONTENT_MATCHES = 2
RETRIEVAL_AUDIT_MIN_TOP_CONTENT_COVERAGE = 0.10

LEGACY_SYNONYMS = {
    "appointments": "appointment",
    "appointment": "appointment",
    "booking": "booking",
    "bookings": "booking",
    "scheduled": "schedule",
    "scheduling": "schedule",
    "doctors": "doctor",
    "clinicians": "clinician",
    "patients": "patient",
    "receptionists": "receptionist",
    "databases": "database",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "apis": "api",
    "services": "service",
    "requests": "request",
    "records": "record",
    "cancellations": "cancellation",
    "cancelled": "cancel",
    "canceled": "cancel",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def simple_stem(token: str) -> str:
    token = LEGACY_SYNONYMS.get(token, token)
    for suffix in ("ization", "ations", "ation", "ingly", "edly", "ing", "ers", "ies", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            token = token[: -len(suffix)]
            break
    return token


def tokenize(text: str) -> list[str]:
    """Conservative Unicode tokenizer used by production retrieval.

    This intentionally preserves Vietnamese accents and does not invent
    synonyms or perform crude stemming. It is a deterministic lexical
    normalization, not a semantic query expansion step.
    """
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", normalized, flags=re.UNICODE)


def legacy_tokenize(text: str) -> list[str]:
    """Compatibility tokenizer for the previous TF-IDF/debug path only."""
    expanded = text.lower()
    for source, replacement in LEGACY_VIETNAMESE_EXPANSIONS.items():
        expanded = expanded.replace(source, f" {replacement} ")
    expanded = ascii_fold(expanded)
    raw_tokens = re.findall(r"[a-z][a-z0-9]+", expanded)
    return [simple_stem(token) for token in raw_tokens if token not in LEGACY_STOPWORDS and len(token) > 1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_ref(locator: str = "input.raw_text") -> dict[str, str]:
    return {"source_id": "SRC-INPUT-0001", "locator": locator}


def _requirement_id(prefix: str, index: int) -> str:
    return f"{prefix}-{index:03d}"


def _parse_labeled_lines(raw_text: str) -> tuple[list[str], list[str], list[str]]:
    functional: list[str] = []
    non_functional: list[str] = []
    constraints: list[str] = []
    for line in raw_text.splitlines():
        line = line.strip().strip("-•")
        if not line:
            continue
        normalized = line.lower()
        if re.match(r"^(nfr|non[- ]functional|performance|security)\b", normalized):
            non_functional.append(re.sub(r"^[^:]+:\s*", "", line))
        elif re.match(r"^(c\d*|constraint|ràng buộc)\b", normalized):
            constraints.append(re.sub(r"^[^:]+:\s*", "", line))
        elif re.match(r"^(fr|r\d+|functional|use case)\b", normalized):
            functional.append(re.sub(r"^[^:]+:\s*", "", line))
    return functional, non_functional, constraints


def normalize_input(data: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(data.get("raw_text", "")).strip()
    if not raw_text:
        raise ValueError("input.raw_text must not be empty")

    parsed_functional, parsed_nfr, parsed_constraints = _parse_labeled_lines(raw_text)
    actors = data.get("actors") or []
    if actors and isinstance(actors[0], str):
        actors = [{"name": actor, "description": f"Actor mentioned by the stakeholder: {actor}."} for actor in actors]
    if not actors:
        actor_match = re.search(r"(?:actors?|người dùng|vai trò)\s*:\s*([^\n.]+)", raw_text, re.IGNORECASE)
        if actor_match:
            names = re.split(r",|\band\b|\bvà\b", actor_match.group(1))
            actors = [{"name": name.strip(), "description": f"Actor mentioned by the stakeholder: {name.strip()}."} for name in names if name.strip()]
    if not actors:
        actors = [{"name": "User", "description": "A user interacting with the system."}]

    problem = str(data.get("problem_statement") or raw_text.splitlines()[0])
    summary = str(data.get("system_summary") or raw_text.replace("\n", " "))
    functional_items = data.get("functional_requirements") or [
        {"id": _requirement_id("FR", i), "statement": value, "priority": "must"}
        for i, value in enumerate(parsed_functional or [summary], 1)
    ]
    nfr_items = data.get("non_functional_requirements") or [
        {"id": _requirement_id("NFR", i), "statement": value, "category": "not_specified", "priority": "should"}
        for i, value in enumerate(parsed_nfr, 1)
    ]
    constraint_items = data.get("constraints") or [
        {"id": _requirement_id("C", i), "statement": value}
        for i, value in enumerate(parsed_constraints, 1)
    ]

    def enrich(items: Iterable[dict[str, Any]], default_prefix: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for index, item in enumerate(items, 1):
            item = dict(item)
            item.setdefault("id", _requirement_id(default_prefix, index))
            item.setdefault("priority", "must" if default_prefix == "FR" else "should")
            item.setdefault("source_refs", [_source_ref()])
            item.setdefault("rag_eligible", True)
            item.setdefault("leakage", {"classification": "none", "action": "keep", "rationale": "Stage 1 stakeholder input."})
            output.append(item)
        return output

    actor_items = []
    for index, actor in enumerate(actors, 1):
        if isinstance(actor, str):
            actor = {"name": actor}
        actor_items.append({
            "id": actor.get("id", f"ACT-{index:03d}"),
            "name": actor.get("name", f"Actor {index}"),
            "description": actor.get("description", "Actor mentioned by the stakeholder."),
            "source_refs": actor.get("source_refs", [_source_ref("input.actors")]),
        })

    return {
        "schema_version": MOCK_SCHEMA_VERSION,
        "case_id": data.get("case_id", "case-900001"),
        "canonical_language": data.get("canonical_language", "vi"),
        "status": "normalized",
        "raw_text": raw_text,
        "problem_statement": problem,
        "system_summary": summary,
        "actors": actor_items,
        "functional_requirements": enrich(functional_items, "FR"),
        "non_functional_requirements": enrich(nfr_items, "NFR"),
        "constraints": enrich(constraint_items, "C"),
        "architecture_mentions": data.get("architecture_mentions", []),
        "domain_terms": data.get("domain_terms", []),
        "architecture_hints": data.get("architecture_hints", {}),
        "extensions": {
            "stage": "S02",
            "input_artifact": "input.json",
            "claim_boundary": "normalized stakeholder input; not a human-approved architecture",
        },
    }


def _cosine(query_tf: dict[str, int], doc_tf: dict[str, int], idf: dict[str, float]) -> float:
    common = set(query_tf) & set(doc_tf) & set(idf)
    if not common:
        return 0.0

    def weight(term: str, tf: int) -> float:
        return (1.0 + math.log(tf)) * idf.get(term, 0.0)

    q_norm = math.sqrt(sum(weight(term, tf) ** 2 for term, tf in query_tf.items() if term in idf))
    d_norm = math.sqrt(sum(weight(term, tf) ** 2 for term, tf in doc_tf.items() if term in idf))
    if not q_norm or not d_norm:
        return 0.0
    return sum(weight(term, query_tf[term]) * weight(term, doc_tf[term]) for term in common) / (q_norm * d_norm)


@dataclass(frozen=True)
class RetrievalConfig:
    """Explicit retrieval configuration.

    The defaults are engineering starting points. They are not claimed to be
    optimal for this KB because the repository has no labelled relevance set.
    """

    k1: float = 1.2
    rrf_k: int = 60
    candidate_multiplier: int = 5
    field_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_FIELD_WEIGHTS))
    field_b: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_FIELD_B))


class EmbeddingProvider(Protocol):
    model_name: str
    vectors_are_l2_normalized: bool

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        ...


class SentenceTransformerEmbeddingProvider:
    """Optional dense bi-encoder adapter; importing it is deferred."""

    vectors_are_l2_normalized = True

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        *,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on local extras
            raise RuntimeError(
                "Semantic retrieval requires the optional sentence-transformers dependency."
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self._model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]


class SentenceTransformerCrossEncoderReranker:
    """Optional multilingual cross-encoder adapter; scores are ranking-only."""

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-reranker-base",
        *,
        trust_remote_code: bool = False,
        batch_size: int = 16,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on local extras
            raise RuntimeError(
                "Reranking requires the optional sentence-transformers dependency."
            ) from exc
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, trust_remote_code=trust_remote_code)
        self.score_transform = type(self._model.activation_fn).__name__.lower()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        pairs = [[query, document] for document in documents]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return [float(value) for value in scores]


def _field_values(projection: Mapping[str, Any]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for field_name in RETRIEVAL_FIELDS:
        raw = projection.get(field_name, [])
        if isinstance(raw, list):
            values[field_name] = [str(item) for item in raw if str(item).strip()]
        elif raw is None:
            values[field_name] = []
        else:
            values[field_name] = [str(raw)]
    return values


def _unit_fields(field_name: str, text: str) -> dict[str, str]:
    return {name: text if name == field_name else "" for name in RETRIEVAL_FIELDS}


def _source_locator(item: Mapping[str, Any], fallback: str) -> str:
    refs = item.get("source_refs") or []
    if refs and isinstance(refs[0], Mapping):
        return str(refs[0].get("locator") or fallback)
    return fallback


def _query_eligible(item: Mapping[str, Any]) -> bool:
    if item.get("rag_eligible") is not True:
        return False
    leakage = item.get("leakage") or {}
    if leakage.get("action") in {"exclude_from_rag", "drop", "remove"}:
        return False
    if leakage.get("classification") in {"solution_leakage", "known_target_architecture", "ground_truth_answer"}:
        return False
    return True


def build_retrieval_queries(requirements: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create deterministic, source-preserving queries without LLM expansion."""
    queries: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    context_parts = [str(requirements.get("problem_statement", "")).strip()]
    context_parts.extend(
        str(actor.get("description", "")).strip()
        for actor in requirements.get("actors", [])
        if str(actor.get("description", "")).strip()
    )
    context_text = " ".join(part for part in context_parts if part)
    if context_text:
        queries.append({
            "query_id": "CONTEXT-001",
            "requirement_id": "CONTEXT-001",
            "query_kind": "architecture_neutral_context",
            "query_text": context_text,
            "source_refs": [_source_ref("input.problem_statement")],
            "rag_eligible": True,
            "leakage": {"classification": "none", "action": "keep"},
        })

    for group, prefix in (
        ("functional_requirements", "FR"),
        ("non_functional_requirements", "NFR"),
        ("constraints", "C"),
    ):
        for index, item in enumerate(requirements.get(group, []), 1):
            requirement_id = str(item.get("id") or f"{prefix}-{index:03d}")
            if not _query_eligible(item):
                exclusions.append({
                    "requirement_id": requirement_id,
                    "reason": "rag_eligible=false or leakage policy excludes this item",
                    "leakage": item.get("leakage"),
                })
                continue
            statement = str(item.get("statement", "")).strip()
            if not statement:
                exclusions.append({"requirement_id": requirement_id, "reason": "empty statement"})
                continue
            queries.append({
                "query_id": requirement_id,
                "requirement_id": requirement_id,
                "query_kind": group.removesuffix("_requirements"),
                "query_text": statement,
                "source_refs": item.get("source_refs", []),
                "rag_eligible": True,
                "leakage": item.get("leakage", {}),
            })

    return queries, exclusions


def build_evidence_units(kb_root: Path, index: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build stable evidence units from actual KB requirements and metadata."""
    units: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for document in sorted(index.get("documents", []), key=lambda item: str(item.get("case_id"))):
        case_id = str(document["case_id"])
        case_root = kb_root / "cases" / "kb" / case_id
        requirements = read_json(case_root / "requirements.json")
        case_data = read_json(case_root / "case.json")
        sequence = 1

        def add_unit(
            evidence_type: str,
            field_name: str,
            text: str,
            requirement_id: str | None,
            source_path: str,
            source_locator: str | None,
        ) -> None:
            nonlocal sequence
            text = str(text).strip()
            if not text:
                return
            units.append({
                "evidence_id": f"{case_id}:E{sequence:03d}",
                "case_id": case_id,
                "evidence_type": evidence_type,
                "requirement_id": requirement_id,
                "text": text,
                "field_name": field_name,
                "fields": _unit_fields(field_name, text),
                "source_path": source_path,
                "source_locator": source_locator,
                "review_status": case_data.get("review_status"),
                "evidence_tier": case_data.get("evidence_tier"),
                "system_group_id": document.get("system_group_id"),
                "source_id": document.get("source_id"),
                "provider": document.get("provider"),
                "primary_domain": document.get("primary_domain"),
            })
            sequence += 1

        add_unit(
            "problem_statement",
            "problem_statement",
            requirements.get("problem_statement", ""),
            None,
            f"cases/kb/{case_id}/requirements.json",
            "problem_statement",
        )
        for actor in requirements.get("actors", []):
            add_unit(
                "actor_description",
                "actor_descriptions",
                actor.get("description", ""),
                actor.get("id"),
                f"cases/kb/{case_id}/requirements.json",
                _source_locator(actor, f"actors[{actor.get('id', 'unknown')}].description"),
            )
        for group, evidence_type, field_name in (
            ("functional_requirements", "functional_requirement", "functional_requirements"),
            ("non_functional_requirements", "non_functional_requirement", "non_functional_requirements"),
            ("constraints", "constraint", "eligible_constraints"),
        ):
            for item in requirements.get(group, []):
                if not _query_eligible(item):
                    exclusions.append({
                        "evidence_id": f"{case_id}:{item.get('id', 'unknown')}",
                        "case_id": case_id,
                        "reason": "KB item excluded by rag_eligible/leakage policy",
                        "source_path": f"cases/kb/{case_id}/requirements.json",
                        "source_locator": _source_locator(item, f"{group}[{item.get('id', 'unknown')}]"),
                    })
                    continue
                add_unit(
                    evidence_type,
                    field_name,
                    item.get("statement", ""),
                    item.get("id"),
                    f"cases/kb/{case_id}/requirements.json",
                    _source_locator(item, f"{group}[{item.get('id', 'unknown')}]"),
                )
        domain_tags = case_data.get("domain_tags") or document.get("projection", {}).get("domain_tags", [])
        if domain_tags:
            add_unit(
                "domain_tag",
                "domain_tags",
                " ".join(str(tag) for tag in domain_tags),
                None,
                f"cases/kb/{case_id}/case.json",
                "domain_tags",
            )
    return units, exclusions


def _build_field_index(units: Sequence[Mapping[str, Any]], config: RetrievalConfig) -> dict[str, Any]:
    document_frequency: Counter[str] = Counter()
    documents: list[dict[str, Any]] = []
    length_totals = {field_name: 0 for field_name in RETRIEVAL_FIELDS}
    for unit in units:
        field_counts: dict[str, Counter[str]] = {}
        for field_name, value in (unit.get("fields") or {}).items():
            counts = Counter(tokenize(str(value)))
            field_counts[field_name] = counts
            length_totals[field_name] += sum(counts.values())
            document_frequency.update(counts.keys())
        documents.append({"unit": unit, "field_counts": field_counts})
    count = len(documents) or 1
    average_lengths = {name: total / count for name, total in length_totals.items()}
    idf = {
        term: math.log((count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in sorted(document_frequency.items())
    }
    return {"documents": documents, "document_frequency": dict(document_frequency), "idf": idf, "average_lengths": average_lengths}


def _bm25f_score(
    query_tf: Mapping[str, int],
    field_counts: Mapping[str, Mapping[str, int]],
    average_lengths: Mapping[str, float],
    idf: Mapping[str, float],
    config: RetrievalConfig,
) -> float:
    score = 0.0
    for term, query_frequency in query_tf.items():
        if term not in idf:
            continue
        pseudo_tf = 0.0
        for field_name in RETRIEVAL_FIELDS:
            tf = float(field_counts.get(field_name, {}).get(term, 0))
            if not tf:
                continue
            field_length = float(sum(field_counts.get(field_name, {}).values()))
            average_length = float(average_lengths.get(field_name, 0.0))
            b = float(config.field_b.get(field_name, 0.75))
            normalization = 1.0 if average_length == 0 else (1.0 - b) + b * field_length / average_length
            pseudo_tf += float(config.field_weights.get(field_name, 1.0)) * tf / normalization
        if pseudo_tf:
            saturation = ((config.k1 + 1.0) * pseudo_tf) / (config.k1 + pseudo_tf)
            score += float(query_frequency) * saturation * float(idf[term])
    return score


def reciprocal_rank_fusion(rankings: Mapping[str, Sequence[str]], k: int = 60) -> list[dict[str, Any]]:
    """Return deterministic RRF scores; rank positions are one-based."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: defaultdict[str, float] = defaultdict(float)
    rank_by_item: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for ranker_name, ranking in rankings.items():
        for rank, item_id in enumerate(ranking, 1):
            if ranker_name in rank_by_item[item_id]:
                raise ValueError(f"Duplicate item {item_id!r} in ranker {ranker_name!r}")
            rank_by_item[item_id][ranker_name] = rank
            scores[item_id] += 1.0 / (k + rank)
    fused = [
        {"item_id": item_id, "rrf_score": score, "ranks": rank_by_item[item_id]}
        for item_id, score in scores.items()
    ]
    fused.sort(key=lambda item: (-item["rrf_score"], item["item_id"]))
    for rank, item in enumerate(fused, 1):
        item["rrf_rank"] = rank
    return fused


def _load_case_pattern(kb_root: Path, case_id: str) -> dict[str, Any]:
    case_root = kb_root / "cases" / "kb" / case_id
    case_data = read_json(case_root / "case.json") if (case_root / "case.json").exists() else {}
    component_text = (case_root / "component.puml").read_text(encoding="utf-8") if (case_root / "component.puml").exists() else ""
    deployment_text = (case_root / "deployment.puml").read_text(encoding="utf-8") if (case_root / "deployment.puml").exists() else ""

    logical_re = re.compile(r'^\s*(actor|component|database|queue|storage|package)\s+"([^"]+)"\s+as\s+([a-z][a-z0-9_]*)\s*(?:<<([^>]+)>>)?')
    deployment_re = re.compile(r'^\s*(frame|node|cloud|database|queue|storage|artifact)\s+"([^"]+)"\s+as\s+([a-z][a-z0-9_]*)\s*(?:<<([^>]+)>>)?')
    relation_re = re.compile(r'^\s*([a-z][a-z0-9_]*)\s+[-.]+>\s+([a-z][a-z0-9_]*)\s*:\s*\[([^\]]+)\]\s*([A-Z_]+)')

    logical = []
    technologies: set[str] = set()
    logical_kinds: set[str] = set()
    for line in component_text.splitlines():
        match = logical_re.match(line)
        if not match:
            continue
        keyword, name, alias, stereotype = match.groups()
        tokens = [token.strip() for token in (stereotype or "").split(",")]
        logical.append({"keyword": keyword, "name": name, "alias": alias, "stereotypes": tokens})
        technologies.update(token.removeprefix("tech.") for token in tokens if token.startswith("tech."))
        logical_kinds.update(token.removeprefix("kind.") for token in tokens if token.startswith("kind."))

    deployment = []
    node_kinds: set[str] = set()
    environments: set[str] = set()
    for line in deployment_text.splitlines():
        match = deployment_re.match(line)
        if not match:
            continue
        keyword, name, alias, stereotype = match.groups()
        tokens = [token.strip() for token in (stereotype or "").split(",")]
        deployment.append({"keyword": keyword, "name": name, "alias": alias, "stereotypes": tokens})
        node_kinds.update(token.removeprefix("node.") for token in tokens if token.startswith("node."))
        environments.update(token.removeprefix("environment.") for token in tokens if token.startswith("environment."))

    relations = []
    for line in component_text.splitlines() + deployment_text.splitlines():
        match = relation_re.match(line)
        if match:
            relations.append({"source": match.group(1), "destination": match.group(2), "id": match.group(3), "kind": match.group(4)})

    return {
        "case_id": case_id,
        "title": case_data.get("title", case_id),
        "source_ids": case_data.get("source_ids", []),
        "evidence_tier": case_data.get("evidence_tier"),
        "review_status": case_data.get("review_status"),
        "component_patterns": {
            "element_count": len(logical),
            "logical_kinds": sorted(logical_kinds),
            "technologies": sorted(technologies),
            "relation_kinds": sorted({relation["kind"] for relation in relations if relation["kind"] not in {"NETWORK"}}),
        },
        "deployment_patterns": {
            "element_count": len(deployment),
            "node_kinds": sorted(node_kinds),
            "environments": sorted(environments),
            "network_relation_count": sum(1 for relation in relations if relation["kind"] == "NETWORK"),
        },
        "source_paths": [
            f"cases/kb/{case_id}/requirements.json",
            f"cases/kb/{case_id}/component.puml",
            f"cases/kb/{case_id}/deployment.puml",
        ],
    }


def retrieve(
    kb_root: Path,
    requirements: dict[str, Any],
    top_k: int = 3,
    *,
    config: RetrievalConfig | None = None,
    semantic_provider: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
) -> dict[str, Any]:
    """Retrieve source-grounded evidence units for each normalized query."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    config = config or RetrievalConfig()
    index = read_json(kb_root / "derived" / "retrieval" / "production-index.json")
    queries, query_exclusions = build_retrieval_queries(requirements)
    units, evidence_exclusions = build_evidence_units(kb_root, index)
    field_index = _build_field_index(units, config)
    unit_by_id = {str(unit["evidence_id"]): unit for unit in units}
    pattern_cache: dict[str, dict[str, Any]] = {}

    semantic_vectors: dict[str, list[float]] = {}
    semantic_status = "not_configured"
    if semantic_provider is not None:
        encoded = semantic_provider.encode([str(unit["text"]) for unit in units])
        if len(encoded) != len(units):
            raise ValueError("Embedding provider returned a different number of vectors than evidence units")
        semantic_vectors = {str(unit["evidence_id"]): vector for unit, vector in zip(units, encoded)}
        semantic_status = "enabled"

    def similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding dimensions do not match")
        dot = sum(a * b for a, b in zip(left, right))
        if getattr(semantic_provider, "vectors_are_l2_normalized", False):
            return dot
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    query_outputs: list[dict[str, Any]] = []
    query_support_audit: list[dict[str, Any]] = []
    for query in queries:
        query_text = str(query["query_text"])
        query_tf = Counter(tokenize(query_text))
        query_tokens = sorted(query_tf)
        audit_content_terms = sorted(
            term for term in query_tokens
            if term not in RETRIEVAL_AUDIT_GENERIC_TERMS
        )
        known_content_terms = sorted(
            term for term in audit_content_terms
            if term in field_index["idf"]
        )
        oov_content_terms = sorted(set(audit_content_terms) - set(known_content_terms))
        lexical_rows: list[dict[str, Any]] = []
        for document in field_index["documents"]:
            unit = document["unit"]
            lexical_score = _bm25f_score(
                query_tf,
                document["field_counts"],
                field_index["average_lengths"],
                field_index["idf"],
                config,
            )
            matched_terms = sorted(set(query_tf) & {
                term
                for counts in document["field_counts"].values()
                for term in counts
            })
            matched_content_terms = sorted(
                term for term in matched_terms
                if term not in RETRIEVAL_AUDIT_GENERIC_TERMS
            )
            lexical_rows.append({
                "evidence_id": unit["evidence_id"],
                "lexical_score": lexical_score,
                "matched_terms": matched_terms,
                "matched_content_terms": matched_content_terms,
            })
        lexical_rows.sort(key=lambda row: (-row["lexical_score"], row["evidence_id"]))
        lexical_rank = {row["evidence_id"]: rank for rank, row in enumerate(lexical_rows, 1)}
        lexical_by_id = {row["evidence_id"]: row for row in lexical_rows}

        semantic_rows: list[dict[str, Any]] = []
        semantic_rank: dict[str, int] = {}
        semantic_by_id: dict[str, dict[str, Any]] = {}
        if semantic_provider is not None:
            query_vector = semantic_provider.encode([query_text])[0]
            semantic_rows = [
                {
                    "evidence_id": evidence_id,
                    "semantic_score": similarity(query_vector, vector),
                }
                for evidence_id, vector in semantic_vectors.items()
            ]
            semantic_rows.sort(key=lambda row: (-row["semantic_score"], row["evidence_id"]))
            semantic_rank = {row["evidence_id"]: rank for rank, row in enumerate(semantic_rows, 1)}
            semantic_by_id = {row["evidence_id"]: row for row in semantic_rows}

        rank_lists: dict[str, Sequence[str]] = {
            "lexical": [row["evidence_id"] for row in lexical_rows],
        }
        if semantic_provider is not None:
            rank_lists["semantic"] = [row["evidence_id"] for row in semantic_rows]
        fused = reciprocal_rank_fusion(rank_lists, k=config.rrf_k)
        fused_by_id = {row["item_id"]: row for row in fused}
        candidate_count = min(len(fused), max(top_k, top_k * config.candidate_multiplier))
        candidate_ids = [row["item_id"] for row in fused[:candidate_count]]

        reranker_scores: dict[str, float] = {}
        reranker_rank: dict[str, int] = {}
        if reranker is not None:
            scores = reranker.score(query_text, [unit_by_id[item_id]["text"] for item_id in candidate_ids])
            if len(scores) != len(candidate_ids):
                raise ValueError("Reranker returned a different number of scores than candidates")
            reranked = sorted(
                zip(candidate_ids, scores),
                key=lambda item: (-float(item[1]), item[0]),
            )
            reranker_scores = {item_id: float(score) for item_id, score in reranked}
            reranker_rank = {item_id: rank for rank, (item_id, _) in enumerate(reranked, 1)}
            final_ids = [item_id for item_id, _ in reranked[:top_k]]
        else:
            final_ids = candidate_ids[:top_k]

        evidence: list[dict[str, Any]] = []
        final_rank_by_id = {evidence_id: rank for rank, evidence_id in enumerate(final_ids, 1)}
        for rank, evidence_id in enumerate(final_ids, 1):
            unit = unit_by_id[evidence_id]
            case_id = str(unit["case_id"])
            if case_id not in pattern_cache:
                pattern_cache[case_id] = _load_case_pattern(kb_root, case_id)
            fused_row = fused_by_id[evidence_id]
            evidence.append({
                "rank": rank,
                "evidence_id": evidence_id,
                "requirement_id": unit.get("requirement_id"),
                "case_id": case_id,
                "evidence_type": unit["evidence_type"],
                "text": unit["text"],
                "lexical_rank": lexical_rank[evidence_id],
                "lexical_score": round(float(lexical_by_id[evidence_id]["lexical_score"]), 12),
                "semantic_rank": semantic_rank.get(evidence_id),
                "semantic_score": (
                    round(float(semantic_by_id[evidence_id]["semantic_score"]), 12)
                    if evidence_id in semantic_by_id else None
                ),
                "rrf_rank": fused_row["rrf_rank"],
                "rrf_score": round(float(fused_row["rrf_score"]), 12),
                "reranker_rank": reranker_rank.get(evidence_id),
                "reranker_score": reranker_scores.get(evidence_id),
                "matched_terms": lexical_by_id[evidence_id]["matched_terms"],
                "matched_content_terms": lexical_by_id[evidence_id]["matched_content_terms"],
                "source_path": unit["source_path"],
                "source_locator": unit["source_locator"],
                "review_status": unit["review_status"],
                "evidence_tier": unit["evidence_tier"],
                "provenance": {
                    "case_id": case_id,
                    "system_group_id": unit["system_group_id"],
                    "source_id": unit["source_id"],
                    "retrieval_basis": "stakeholder requirement query; architecture hints excluded",
                },
            })
        candidate_pool = []
        for evidence_id in candidate_ids:
            unit = unit_by_id[evidence_id]
            fused_row = fused_by_id[evidence_id]
            candidate_pool.append({
                "evidence_id": evidence_id,
                "case_id": unit["case_id"],
                "evidence_type": unit["evidence_type"],
                "text": unit["text"],
                "lexical_rank": lexical_rank[evidence_id],
                "semantic_rank": semantic_rank.get(evidence_id),
                "semantic_score": (
                    round(float(semantic_by_id[evidence_id]["semantic_score"]), 12)
                    if evidence_id in semantic_by_id else None
                ),
                "rrf_rank": fused_row["rrf_rank"],
                "rrf_score": round(float(fused_row["rrf_score"]), 12),
                "reranker_rank": reranker_rank.get(evidence_id),
                "reranker_score": reranker_scores.get(evidence_id),
                "final_rank": final_rank_by_id.get(evidence_id),
                "source_path": unit["source_path"],
                "source_locator": unit["source_locator"],
            })
        candidate_pool.sort(
            key=lambda row: (
                row["reranker_rank"] if reranker is not None else row["rrf_rank"],
                row["evidence_id"],
            )
        )
        top_evidence = evidence[0] if evidence else None
        top_content_terms = top_evidence["matched_content_terms"] if top_evidence else []
        if not audit_content_terms:
            support_status = "insufficient_content_terms"
        elif (
            len(known_content_terms) >= RETRIEVAL_AUDIT_MIN_KNOWN_CONTENT_TERMS
            and (
                len(top_content_terms) >= RETRIEVAL_AUDIT_MIN_TOP_CONTENT_MATCHES
                or len(top_content_terms) / max(len(audit_content_terms), 1)
                >= RETRIEVAL_AUDIT_MIN_TOP_CONTENT_COVERAGE
            )
        ):
            support_status = "lexically_supported"
        else:
            support_status = "weak_lexical_support"
        query_support_audit.append({
            "query_id": query["query_id"],
            "content_query_terms": audit_content_terms,
            "known_content_terms": known_content_terms,
            "oov_content_terms": oov_content_terms,
            "top_evidence_id": top_evidence["evidence_id"] if top_evidence else None,
            "top_matched_content_terms": top_content_terms,
            "content_coverage": round(
                len(top_content_terms) / max(len(audit_content_terms), 1),
                12,
            ),
            "support_status": support_status,
        })
        query_outputs.append({
            "query_id": query["query_id"],
            "requirement_id": query["requirement_id"],
            "query_kind": query["query_kind"],
            "query_text": query_text,
            "source_refs": query["source_refs"],
            "rag_eligible": query["rag_eligible"],
            "leakage": query["leakage"],
            "query_tokens": query_tokens,
            "audit_content_terms": audit_content_terms,
            "audit_known_content_terms": known_content_terms,
            "audit_oov_content_terms": oov_content_terms,
            "support_status": support_status,
            "candidate_pool_size": candidate_count,
            "candidate_pool": candidate_pool,
            "evidence": evidence,
        })

    weak_query_ids = [
        item["query_id"]
        for item in query_support_audit
        if item["support_status"] != "lexically_supported"
    ]
    if not query_support_audit or len(weak_query_ids) == len(query_support_audit):
        audit_overall_status = "out_of_domain_candidate"
    elif weak_query_ids:
        audit_overall_status = "mixed_lexical_support"
    else:
        audit_overall_status = "lexically_supported"

    case_rows: dict[str, dict[str, Any]] = {}
    for query_output in query_outputs:
        for evidence in query_output["evidence"]:
            case_id = evidence["case_id"]
            if case_id in case_rows:
                if evidence["evidence_id"] not in case_rows[case_id]["evidence_ids"]:
                    case_rows[case_id]["evidence_ids"].append(evidence["evidence_id"])
                case_rows[case_id]["best_final_rank"] = min(case_rows[case_id]["best_final_rank"], evidence["rank"])
                case_rows[case_id]["best_rrf_rank"] = min(case_rows[case_id]["best_rrf_rank"], evidence["rrf_rank"])
                case_rows[case_id]["best_rrf_score"] = max(case_rows[case_id]["best_rrf_score"], evidence["rrf_score"])
                case_rows[case_id]["best_lexical_score"] = max(case_rows[case_id]["best_lexical_score"], evidence["lexical_score"])
                if evidence["reranker_score"] is not None:
                    current = case_rows[case_id]["best_reranker_score"]
                    case_rows[case_id]["best_reranker_score"] = max(
                        current if current is not None else float("-inf"),
                        evidence["reranker_score"],
                    )
                continue
            pattern = pattern_cache[case_id]
            case_rows[case_id] = {
                "rank": 0,
                "case_id": case_id,
                "best_final_rank": evidence["rank"],
                "best_rrf_rank": evidence["rrf_rank"],
                "best_rrf_score": evidence["rrf_score"],
                "best_lexical_score": evidence["lexical_score"],
                "best_reranker_score": evidence["reranker_score"],
                "evidence_ids": [evidence["evidence_id"]],
                "component_patterns": pattern["component_patterns"],
                "deployment_patterns": pattern["deployment_patterns"],
                "source_paths": pattern["source_paths"],
                "evidence_tier": pattern["evidence_tier"],
                "review_status": pattern["review_status"],
            }

    if semantic_provider is not None:
        semantic_method = {
            "name": "dense_embedding",
            "status": "enabled",
            "model": semantic_provider.model_name,
            "vectors_are_l2_normalized": bool(getattr(semantic_provider, "vectors_are_l2_normalized", False)),
            "trust_remote_code": getattr(semantic_provider, "trust_remote_code", False),
            "similarity": "dot_product_of_l2_normalized_vectors" if getattr(semantic_provider, "vectors_are_l2_normalized", False) else "cosine_similarity",
            "score_interpretation": "retrieval similarity score; not a probability or confidence",
        }
    else:
        semantic_method = {
            "name": "dense_embedding",
            "status": semantic_status,
            "model": None,
            "trust_remote_code": False,
            "score_interpretation": "No semantic score was fabricated because no embedding provider was configured.",
        }
    reranker_method = {
        "name": "cross_encoder" if reranker is not None else None,
        "model": reranker.model_name if reranker is not None else None,
        "status": "enabled" if reranker is not None else "not_configured",
        "trust_remote_code": getattr(reranker, "trust_remote_code", False) if reranker is not None else False,
        "candidate_pool": "top-M RRF candidates" if reranker is not None else None,
        "candidate_multiplier": config.candidate_multiplier if reranker is not None else None,
        "score_transform": getattr(reranker, "score_transform", None) if reranker is not None else None,
        "score_interpretation": "ranking score; not a probability or confidence",
    }
    final_ranking_basis = "reranker" if reranker is not None else "rrf"
    ranked_cases = sorted(
        case_rows.values(),
        key=lambda row: (
            row["best_final_rank"],
            -(row["best_reranker_score"] if reranker is not None and row["best_reranker_score"] is not None else row["best_rrf_score"]),
            row["case_id"],
        ),
    )
    for rank, row in enumerate(ranked_cases, 1):
        row["rank"] = rank

    return {
        "schema_version": MOCK_SCHEMA_VERSION,
        "query_case_id": requirements["case_id"],
        "index_id": index["index_id"],
        "index_version": index["schema_version"],
        "index_generated_at": index["generated_at"],
        "method": {
            "lexical": {
                "name": "bm25f",
                "variant": "Robertson-Zaragoza BM25F Eq. 3.19-3.21 with collection-wide RSJ IDF",
                "k1": config.k1,
                "field_weights": dict(config.field_weights),
                "field_b": dict(config.field_b),
                "score_interpretation": "lexical ranking score; not a probability or confidence",
                "legacy_index_field_weights": index.get("field_weights", {}),
                "legacy_weights_classification": "existing project heuristic",
            },
            "semantic": semantic_method,
            "fusion": {
                "name": "rrf",
                "k": config.rrf_k,
                "rank_indexing": "one_based",
                "score_interpretation": "fusion ranking score; not a probability or confidence",
            },
            "reranker": reranker_method,
            "final_ranking": {
                "basis": final_ranking_basis,
                "description": "Evidence rank after reranking when enabled; otherwise RRF rank.",
            },
        },
        "query_policy": {
            "construction": "deterministic one-query-per-normalized-requirement plus a small architecture-neutral context query",
            "llm_query_expansion": False,
            "architecture_hints": "excluded from primary lexical and semantic query text",
            "system_summary": "excluded from primary query text to avoid unclassified solution leakage",
            "leakage_guard": "rag_eligible=true and leakage action/classification must permit retrieval",
        },
        "queries": query_outputs,
        "retrieved_cases": ranked_cases,
        "retrieval_audit": {
            "overall_status": audit_overall_status,
            "weak_query_ids": weak_query_ids,
            "queries": query_support_audit,
            "policy": {
                "generic_terms_excluded": sorted(RETRIEVAL_AUDIT_GENERIC_TERMS),
                "minimum_known_content_terms": RETRIEVAL_AUDIT_MIN_KNOWN_CONTENT_TERMS,
                "minimum_top_content_matches": RETRIEVAL_AUDIT_MIN_TOP_CONTENT_MATCHES,
                "minimum_top_content_coverage": RETRIEVAL_AUDIT_MIN_TOP_CONTENT_COVERAGE,
            },
            "claim_boundary": "Heuristic lexical support audit only; it is not a relevance label, OOD classifier or semantic correctness proof.",
        },
        "exclusions": {
            "input_requirements": query_exclusions,
            "kb_evidence": evidence_exclusions,
        },
        "index_stats": {
            "evidence_unit_count": len(units),
            "document_frequency_terms": len(field_index["idf"]),
            "average_field_lengths": field_index["average_lengths"],
            "field_names": list(RETRIEVAL_FIELDS),
        },
        "claim_boundary": "Retrieval provides source-grounded evidence and precedent, not proof that a new system has the same architecture.",
    }


def _profile_context(kb_root: Path) -> dict[str, Any]:
    profile_path = kb_root / "specs" / "ARCH_KB_PUML_PROFILE_V1_1.md"
    vocabulary_path = kb_root / "vocabulary" / "architecture-vocabulary.json"
    vocabulary = read_json(vocabulary_path)
    relevant_tech = [
        item["key"] for item in vocabulary["technology_keys"]
        if item["key"] in {
            "api.rest", "data.sql", "database.postgresql", "cache.redis", "runtime.dotnet8",
            "framework.fastapi", "deployment.docker_compose", "orchestrator.kubernetes", "net.http",
            "net.https", "net.tcp", "identity.oidc",
        }
    ]
    integrity_checks: list[dict[str, Any]] = []
    dataset_path = kb_root / "dataset.json"
    if dataset_path.exists():
        dataset = read_json(dataset_path)
        profile_decl = dataset.get("puml_profile", {})
        integrity_targets = [
            ("profile", profile_decl.get("path"), profile_decl.get("sha256")),
            ("vocabulary", profile_decl.get("vocabulary_path"), profile_decl.get("vocabulary_sha256")),
            ("vocabulary_schema", profile_decl.get("vocabulary_schema_path"), profile_decl.get("vocabulary_schema_sha256")),
            ("split_manifest", "splits/split-manifest.json", dataset.get("extensions", {}).get("split_manifest_sha256")),
        ]
        for name, relative_path, expected in integrity_targets:
            target = kb_root / relative_path if relative_path else None
            actual = sha256_bytes(target.read_bytes()) if target and target.exists() else None
            integrity_checks.append({
                "name": name,
                "path": relative_path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "pass" if expected and actual == expected else "warning",
            })
    integrity_issues = [
        f"{item['name']}: expected hash does not match current checkout"
        for item in integrity_checks
        if item["status"] != "pass"
    ]
    return {
        "profile_version": PROFILE_VERSION,
        "profile_path": "specs/ARCH_KB_PUML_PROFILE_V1_1.md",
        "profile_sha256": sha256_bytes(profile_path.read_bytes()),
        "vocabulary_version": VOCABULARY_VERSION,
        "vocabulary_path": "vocabulary/architecture-vocabulary.json",
        "vocabulary_sha256": sha256_bytes(vocabulary_path.read_bytes()),
        "integrity_preflight": {
            "status": "warning" if integrity_issues else "pass",
            "checks": integrity_checks,
            "issues": integrity_issues,
            "claim_boundary": "This is a lightweight manifest comparison; the full KB validator remains authoritative.",
        },
        "rules_applied": [
            "one component view and one deployment view share stable aliases",
            "kind, level, scope, role and tech are separate stereotype facts",
            "every logical relation has a unique [rel_*] identifier and typed label",
            "deployment has exactly one outer environment frame",
            "deployment instances use instance-of:<logical_alias>",
            "unknown is preserved as unknown; unsupported concepts are unresolved",
            "no performance number is inferred from a structural diagram",
        ],
        "relevant_vocabulary": relevant_tech,
    }


def build_context(kb_root: Path, requirements: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MOCK_SCHEMA_VERSION,
        "context_id": f"ctx-{requirements['case_id']}",
        "created_at": utc_now(),
        "input": requirements,
        "retrieval": retrieval,
        "profile": _profile_context(kb_root),
        "generation_contract": {
            "goal": "Generate a logically consistent component view and deployment view from requirements plus evidence.",
            "must_preserve": [
                "Only use explicit stakeholder facts or retrieved architectural patterns as evidence.",
                "Use the same logical aliases across component and deployment views.",
                "Every deployed application/data element must point back to one logical element.",
                "Do not invent latency, throughput, CPU demand, availability or benchmark values.",
                "If a protocol, technology or topology is not supported by evidence, use unknown or unresolved.",
            ],
            "output_files": ["architecture-ir.json", "component.puml", "deployment.puml", "validation.json"],
            "human_review_required": True,
        },
    }


def local_llm_ir_schema() -> dict[str, Any]:
    """Small JSON schema passed to Ollama's structured-output API."""
    relation_properties = {
        "relation_id": {"type": "string"},
        "source": {"type": "string"},
        "destination": {"type": "string"},
        "kind": {"type": "string", "enum": ["CALL", "DATA", "PUBLISH", "SUBSCRIBE", "AUTHENTICATE", "AUTHORIZE", "STORAGE", "EXTENSION", "REPLICATE", "USES"]},
        "protocols": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": "string"},
        "access": {"type": "string"}
    }
    component_properties = {
        "alias": {"type": "string"},
        "name": {"type": "string"},
        "kind": {"type": "string"},
        "level": {"type": "string"},
        "scope": {"type": "string"},
        "roles": {"type": "array", "items": {"type": "string"}},
        "technologies": {"type": "array", "items": {"type": "string"}},
        "parent": {"type": "string"},
        "group": {"type": "boolean"}
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "candidate_id", "architecture_style", "rationale",
            "components", "relations", "deployment_environments", "deployment_nodes",
            "deployment_instances", "deployment_relations", "unresolved_concepts",
            "capability_compliance"
        ],
        "properties": {
            "schema_version": {"type": "string"},
            "candidate_id": {"type": "string"},
            "architecture_style": {"type": "string"},
            "rationale": {"type": "string"},
            "components": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["alias", "name", "kind", "level", "scope", "roles", "technologies", "group"], "properties": component_properties}},
            "relations": {"type": "array", "items": {"type": "object", "required": ["relation_id", "source", "destination", "kind", "protocols", "mode", "access"], "properties": relation_properties}},
            "deployment_environments": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["alias", "name", "kind"], "properties": {"alias": {"type": "string"}, "name": {"type": "string"}, "kind": {"type": "string"}}}},
            "deployment_nodes": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["alias", "name", "kind", "technologies"], "properties": {"alias": {"type": "string"}, "name": {"type": "string"}, "kind": {"type": "string"}, "parent": {"type": "string"}, "technologies": {"type": "array", "items": {"type": "string"}}}}},
            "deployment_instances": {"type": "array", "items": {"type": "object", "required": ["instance_alias", "component_alias", "node_alias", "replicas"], "properties": {"instance_alias": {"type": "string"}, "component_alias": {"type": "string"}, "node_alias": {"type": "string"}, "replicas": {"type": "integer", "minimum": 1}}}},
            "deployment_relations": {"type": "array", "items": {"type": "object", "required": ["relation_id", "source_node", "destination_node", "protocols", "logical_relation_id"], "properties": {"relation_id": {"type": "string"}, "source_node": {"type": "string"}, "destination_node": {"type": "string"}, "protocols": {"type": "array", "items": {"type": "string"}}, "logical_relation_id": {"type": "string"}}}},
            "unresolved_concepts": {"type": "array", "items": {"type": "object", "required": ["id", "concept", "reason"], "properties": {"id": {"type": "string"}, "concept": {"type": "string"}, "reason": {"type": "string"}}}},
            "capability_compliance": {"type": "object"}
        }
    }


def _validate_ir_shape(ir: dict[str, Any]) -> None:
    required = {
        "schema_version", "candidate_id", "architecture_style", "rationale", "components",
        "relations", "deployment_environments", "deployment_nodes", "deployment_instances",
        "deployment_relations", "unresolved_concepts", "capability_compliance",
    }
    missing = sorted(required - set(ir))
    if missing:
        raise ValueError(f"Local LLM IR is missing required fields: {', '.join(missing)}")
    if not isinstance(ir["components"], list) or not ir["components"]:
        raise ValueError("Local LLM IR must contain a non-empty components array")
    systems = [item for item in ir["components"] if item.get("group") is True or item.get("kind") == "software_system"]
    if len(systems) != 1:
        raise ValueError("Local LLM IR must contain exactly one software-system group")
    for item in ir["components"]:
        for field in ("alias", "name", "kind", "level", "scope", "roles", "technologies"):
            if field not in item:
                raise ValueError(f"Local LLM component {item.get('alias', '<unknown>')} is missing {field}")
    component_aliases = [item["alias"] for item in ir["components"]]
    if len(component_aliases) != len(set(component_aliases)):
        raise ValueError("Local LLM IR contains duplicate component aliases")
    component_alias_set = set(component_aliases)
    relation_ids = {relation.get("relation_id") for relation in ir["relations"]}
    for relation in ir["relations"]:
        for field in ("relation_id", "source", "destination", "kind", "protocols", "mode", "access"):
            if field not in relation:
                raise ValueError(f"Local LLM relation {relation.get('relation_id', '<unknown>')} is missing {field}")
        if relation.get("source") not in component_alias_set or relation.get("destination") not in component_alias_set:
            raise ValueError(f"Local LLM relation {relation.get('relation_id', '<unknown>')} has an unresolved endpoint")
    environment_aliases = {item.get("alias") for item in ir["deployment_environments"]}
    node_aliases = {item.get("alias") for item in ir["deployment_nodes"]}
    if not environment_aliases or None in environment_aliases or len(environment_aliases) != len(ir["deployment_environments"]):
        raise ValueError("Local LLM IR must contain unique deployment environment aliases")
    if not node_aliases or None in node_aliases or len(node_aliases) != len(ir["deployment_nodes"]):
        raise ValueError("Local LLM IR must contain unique deployment node aliases")
    instance_aliases = [item.get("instance_alias") for item in ir["deployment_instances"]]
    if None in instance_aliases or len(instance_aliases) != len(set(instance_aliases)):
        raise ValueError("Local LLM IR must contain unique deployment instance aliases")
    for instance in ir["deployment_instances"]:
        if instance.get("component_alias") not in component_alias_set:
            raise ValueError(f"Deployment instance {instance.get('instance_alias', '<unknown>')} references an unknown component")
        if instance.get("node_alias") not in node_aliases:
            raise ValueError(f"Deployment instance {instance.get('instance_alias', '<unknown>')} references an unknown node")
    for relation in ir["deployment_relations"]:
        if relation.get("source_node") not in node_aliases or relation.get("destination_node") not in node_aliases:
            raise ValueError(f"Deployment relation {relation.get('relation_id', '<unknown>')} has an unresolved node endpoint")
        if relation.get("logical_relation_id") not in relation_ids:
            raise ValueError(f"Deployment relation {relation.get('relation_id', '<unknown>')} references an unknown logical relation")


def generate_ollama_ir(
    context: dict[str, Any],
    model: str = "qwen3:4b",
    base_url: str = "http://localhost:11434",
    timeout_seconds: int = 300,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Call a local Ollama model and return IR, raw response metadata and stats."""
    prompt = generate_prompt(context) + (
        "\nReturn ONLY one JSON object matching the supplied structured-output schema. "
        "Do not return Markdown, PlantUML, comments, or a reasoning transcript. "
        "Do not invent performance values. Preserve unsupported facts in unresolved_concepts."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": local_llm_ir_schema(),
        "options": {"temperature": 0, "num_ctx": 32768},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code} at {base_url}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. Start Ollama and pull {model}. Details: {exc}"
        ) from exc
    if raw.get("error"):
        raise RuntimeError(f"Ollama returned an error: {raw['error']}")
    text = raw.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    try:
        ir = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama response was not valid JSON: {exc}") from exc
    metadata = {
        "model": raw.get("model", model),
        "done_reason": raw.get("done_reason"),
        "total_duration_ns": raw.get("total_duration"),
        "load_duration_ns": raw.get("load_duration"),
        "prompt_eval_count": raw.get("prompt_eval_count"),
        "eval_count": raw.get("eval_count"),
        "eval_duration_ns": raw.get("eval_duration"),
        "endpoint": base_url.rstrip("/") + "/api/generate",
        "structured_output": True,
    }
    return ir, raw, metadata


def _all_text(context: dict[str, Any]) -> str:
    return json.dumps(context["input"], ensure_ascii=False).lower()


def _tech(hints: dict[str, Any], key: str, text: str, words: tuple[str, ...]) -> bool:
    values = {str(value) for value in hints.values()}
    return key in values or any(word in text for word in words)


def generate_mock_ir(context: dict[str, Any]) -> dict[str, Any]:
    requirements = context["input"]
    text = _all_text(context)
    hints = requirements.get("architecture_hints", {})
    is_clinic = any(re.search(rf"\b{re.escape(word)}\b", text) for word in ("appointment", "clinic", "doctor", "booking"))
    style = "Modular Monolith" if is_clinic or "modular monolith" in text else "Evidence-constrained baseline"
    api_alias = "cmp_appointment_api" if is_clinic else "cmp_application_api"
    db_alias = "dat_appointment_db" if is_clinic else "dat_primary_store"
    cache_alias = "dat_cache"
    system_alias = "sys_clinic_appointment" if is_clinic else "sys_system_of_interest"

    has_db = _tech(hints, "database.postgresql", text, ("postgres", "postgresql", "database", "relational"))
    has_cache = _tech(hints, "cache.redis", text, ("redis", "cache", "bộ nhớ đệm"))
    has_rest = _tech(hints, "api.rest", text, ("rest", "http", "api", "endpoint"))
    has_sql = _tech(hints, "data.sql", text, ("sql", "postgres", "database", "relational"))
    has_dotnet = _tech(hints, "runtime.dotnet8", text, (".net", "dotnet", "asp.net"))
    has_fastapi = _tech(hints, "framework.fastapi", text, ("fastapi",))
    use_compose = _tech(hints, "deployment.docker_compose", text, ("docker compose", "compose", "container"))
    use_k8s = _tech(hints, "orchestrator.kubernetes", text, ("kubernetes", "k8s"))

    elements: list[dict[str, Any]] = []
    for actor in requirements["actors"]:
        alias = "act_" + re.sub(r"[^a-z0-9]+", "_", ascii_fold(actor["name"].lower())).strip("_")
        elements.append({"alias": alias or "act_user", "name": actor["name"], "kind": "actor", "level": "c4_person", "scope": "external", "roles": [], "technologies": []})
    elements.append({
        "alias": system_alias, "name": "Clinic Appointment System" if is_clinic else "System of Interest",
        "kind": "software_system", "level": "c4_software_system", "scope": "system_of_interest",
        "roles": ["system_of_interest"], "technologies": [], "group": True,
    })
    api_tech = []
    if has_dotnet:
        api_tech.append("runtime.dotnet8")
    if has_fastapi:
        api_tech.append("framework.fastapi")
    elements.append({
        "alias": api_alias, "name": "Appointment API" if is_clinic else "Application API",
        "kind": "deployable_unit", "level": "c4_container", "scope": "system_of_interest",
        "roles": ["backend", "http_api"], "technologies": api_tech, "parent": system_alias,
    })
    if has_db:
        elements.append({
            "alias": db_alias, "name": "Appointment Database" if is_clinic else "Primary Database",
            "kind": "data_store", "level": "c4_container", "scope": "system_of_interest",
            "roles": ["database"], "technologies":["database.postgresql"], "parent": system_alias,
        })
    if has_cache:
        elements.append({
            "alias": cache_alias, "name": "Redis Cache", "kind": "cache", "level": "c4_container",
            "scope": "system_of_interest", "roles": ["cache_primary"], "technologies":["cache.redis"], "parent": system_alias,
        })

    relations: list[dict[str, Any]] = []
    actors = [element for element in elements if element["kind"] == "actor"]
    for index, actor in enumerate(actors, 1):
        relations.append({"relation_id": f"rel_actor_api_{index:02d}", "source": actor["alias"], "destination": api_alias, "kind": "CALL", "protocols": ["api.rest"] if has_rest else ["unknown"], "mode": "synchronous" if has_rest else "unknown", "access": "unknown"})
    if has_db:
        relations.append({"relation_id": "rel_api_database", "source": api_alias, "destination": db_alias, "kind": "DATA", "access": "read_write", "protocols": ["data.sql"] if has_sql else ["unknown"]})
    if has_cache:
        relations.append({"relation_id": "rel_api_cache", "source": api_alias, "destination": cache_alias, "kind": "DATA", "access": "read_write", "protocols": ["net.tcp"]})

    env_alias = "env_production" if "production" in text else "env_local"
    env_name = "Production" if env_alias == "env_production" else "Local"
    nodes: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    deployment_relations: list[dict[str, Any]] = []
    if use_k8s:
        nodes.extend([
            {"alias":"node_cluster", "name":"Kubernetes Cluster", "kind":"orchestration_cluster", "technologies":["orchestrator.kubernetes"]},
            {"alias":"node_api", "name":"API Workload", "kind":"container", "parent":"node_cluster", "technologies":[]},
        ])
    else:
        nodes.append({"alias":"node_app", "name":"Application Host", "kind":"docker_host" if use_compose else "virtual_machine", "technologies":["deployment.docker_compose"] if use_compose else []})
    api_node = "node_api" if use_k8s else "node_app"
    instances.append({"instance_alias":"inst_api", "component_alias":api_alias, "node_alias":api_node, "replicas":1})
    if has_db:
        db_node = "node_db"
        nodes.append({"alias":db_node, "name":"PostgreSQL Database", "kind":"container" if use_compose else "managed_database_service", "technologies":["database.postgresql"]})
        instances.append({"instance_alias":"inst_database", "component_alias":db_alias, "node_alias":db_node, "replicas":1})
        deployment_relations.append({"relation_id":"rel_net_api_database", "source_node":api_node, "destination_node":db_node, "protocols":["net.tcp"], "logical_relation_id":"rel_api_database"})
    if has_cache:
        cache_node = "node_cache"
        nodes.append({"alias":cache_node, "name":"Redis Cache", "kind":"container" if use_compose else "managed_service", "technologies":["cache.redis"]})
        instances.append({"instance_alias":"inst_cache", "component_alias":cache_alias, "node_alias":cache_node, "replicas":1})
        deployment_relations.append({"relation_id":"rel_net_api_cache", "source_node":api_node, "destination_node":cache_node, "protocols":["net.tcp"], "logical_relation_id":"rel_api_cache"})

    # This is intentionally a small deterministic mock, not an LLM quality claim.
    unresolved = []
    if not has_db:
        unresolved.append({"id":"unres_database", "concept":"persistence technology", "reason":"No database technology was explicit in input or retrieved evidence."})
    return {
        "schema_version": MOCK_SCHEMA_VERSION,
        "candidate_id": f"candidate-{requirements['case_id']}",
        "architecture_style": style,
        "rationale": "Generated from normalized requirements, retrieved structural patterns and the pinned ARCH-KB-PUML constraints. The rationale is a candidate explanation, not a production architecture decision.",
        "components": elements,
        "relations": relations,
        "deployment_environments": [{"alias":env_alias, "name":env_name, "kind":"production" if env_alias == "env_production" else "local"}],
        "deployment_nodes": nodes,
        "deployment_instances": instances,
        "deployment_relations": deployment_relations,
        "unresolved_concepts": unresolved,
        "capability_compliance": {
            "profile": PROFILE_VERSION,
            "stage2_adapter": None,
            "unsupported_features": ["performance simulation and prototype execution are not implemented by this mock"],
        },
    }


def _stereotypes(element: dict[str, Any]) -> str:
    tokens = [f"kind.{element['kind']}", f"level.{element['level']}", f"scope.{element['scope']}"]
    tokens.extend(f"role.{role}" for role in element.get("roles", []))
    tokens.extend(f"tech.{tech}" for tech in element.get("technologies", []))
    return "<<" + ",".join(tokens) + ">>"


def _puml_relation(relation: dict[str, Any], source: str, destination: str) -> str:
    relation_id = relation["relation_id"]
    kind = relation["kind"]
    if kind == "CALL":
        return f'{source} --> {destination} : [{relation_id}] CALL(protocols={"|".join(relation["protocols"])},mode={relation["mode"]})'
    if kind == "DATA":
        return f'{source} --> {destination} : [{relation_id}] DATA(access={relation["access"]},protocols={"|".join(relation["protocols"])})'
    return f'{source} --> {destination} : [{relation_id}] USES'


def generate_component_puml(ir: dict[str, Any]) -> str:
    lines = ["@startuml component_view", f"title {ir['architecture_style']} — component view", "skinparam shadowing false", "skinparam componentStyle rectangle", "left to right direction", ""]
    elements = ir["components"]
    actors = [element for element in elements if element["kind"] == "actor"]
    system = next(element for element in elements if element.get("group"))
    for actor in actors:
        lines.append(f'actor "{actor["name"]}" as {actor["alias"]} {_stereotypes(actor)}')
    inner = [element for element in elements if element not in actors and element is not system]
    lines.append(f'package "{system["name"]}" as {system["alias"]} {_stereotypes(system)} {{')
    for element in inner:
        keyword = "database" if element["kind"] in {"data_store", "cache"} else "component"
        lines.append(f'  {keyword} "{element["name"]}" as {element["alias"]} {_stereotypes(element)}')
    lines.append("}")
    lines.append("")
    for relation in ir["relations"]:
        lines.append(_puml_relation(relation, relation["source"], relation["destination"]))
    lines.extend(["", "@enduml", ""])
    return "\n".join(lines)


def generate_deployment_puml(ir: dict[str, Any]) -> str:
    environment = ir["deployment_environments"][0]
    lines = ["@startuml deployment_view", f"title {ir['architecture_style']} — deployment view", "skinparam shadowing false", "left to right direction", "", f'frame "{environment["name"]}" as {environment["alias"]} <<environment.{environment["kind"]}>> {{']
    for node in ir["deployment_nodes"]:
        kind = node["kind"]
        keyword = "cloud" if kind == "cloud_environment" else "node"
        tech = ",".join(f"tech.{tech}" for tech in node.get("technologies", []))
        stereotype = f"<<node.{kind}" + (f",{tech}" if tech else "") + ">>"
        lines.append(f'  {keyword} "{node["name"]}" as {node["alias"]} {stereotype} {{')
        for instance in [item for item in ir["deployment_instances"] if item["node_alias"] == node["alias"]]:
            component = next(item for item in ir["components"] if item["alias"] == instance["component_alias"])
            lines.append(f'    artifact "{component["name"]}" as {instance["instance_alias"]} <<instance-of:{component["alias"]}>>')
        lines.append("  }")
    for relation in ir["deployment_relations"]:
        lines.append(f'  {relation["source_node"]} --> {relation["destination_node"]} : [{relation["relation_id"]}] NETWORK(protocols={"|".join(relation["protocols"])},logical={relation["logical_relation_id"]})')
    lines.extend(["}", "", "@enduml", ""])
    return "\n".join(lines)


def validate_candidate(
    ir: dict[str, Any],
    component_puml: str,
    deployment_puml: str,
    retrieval_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    elements = {element["alias"]: element for element in ir["components"]}
    aliases_unique = len(elements) == len(ir["components"])
    checks.append({"check_id":"CHK-001", "result":"pass" if aliases_unique else "fail", "description":"Logical aliases are unique."})
    relation_endpoints = all(relation["source"] in elements and relation["destination"] in elements for relation in ir["relations"])
    checks.append({"check_id":"CHK-002", "result":"pass" if relation_endpoints else "fail", "description":"All logical relation endpoints resolve."})
    relation_ids = [relation["relation_id"] for relation in ir["relations"]]
    checks.append({"check_id":"CHK-003", "result":"pass" if len(relation_ids) == len(set(relation_ids)) else "fail", "description":"Logical relation IDs are unique."})
    system_aliases = {
        element["alias"]
        for element in ir["components"]
        if element.get("group") is True or element.get("kind") == "software_system"
    }
    instance_refs = all(
        item["component_alias"] in elements and item["component_alias"] not in system_aliases
        for item in ir["deployment_instances"]
    )
    allocated = {item["component_alias"] for item in ir["deployment_instances"]}
    required_allocated = {element["alias"] for element in ir["components"] if element["kind"] not in {"actor", "software_system"}}
    allocation_ok = instance_refs and required_allocated == allocated
    checks.append({"check_id":"CHK-004", "result":"pass" if allocation_ok else "fail", "description":"Every deployable/data logical element has one deployment allocation."})
    checks.append({"check_id":"CHK-005", "result":"pass" if deployment_puml.count("frame ") == 1 else "fail", "description":"Deployment view has exactly one environment frame."})
    relation_labels_present = all(f"[{relation['relation_id']}]" in component_puml for relation in ir["relations"])
    checks.append({"check_id":"CHK-006", "result":"pass" if relation_labels_present and "instance-of:" in deployment_puml else "fail", "description":"PlantUML contains typed relation IDs and instance-of allocations."})
    checks.append({"check_id":"CHK-007", "result":"warning", "description":"No human review or PlantUML renderer execution is claimed by this prototype."})
    if retrieval_audit and retrieval_audit.get("overall_status") == "out_of_domain_candidate":
        checks.append({
            "check_id": "CHK-008",
            "result": "fail",
            "description": "Retrieval audit flags every query as weak lexical support; the candidate is outside the observed KB domain.",
        })
    failed = [check for check in checks if check["result"] == "fail"]
    return {
        "schema_version": MOCK_SCHEMA_VERSION,
        "candidate_id": ir["candidate_id"],
        "profile_version": PROFILE_VERSION,
        "overall_status": "invalid" if failed else "ready_for_review",
        "checks": checks,
        "unresolved_concepts": ir["unresolved_concepts"],
        "retrieval_audit_status": retrieval_audit.get("overall_status") if retrieval_audit else None,
        "claim_boundary": "Static consistency only; not proof of semantic correctness, runtime behavior, performance or production suitability.",
    }


def generate_prompt(context: dict[str, Any]) -> str:
    contract = context["generation_contract"]
    lines = [
        "# Grounded architecture-view generation prompt",
        "",
        "You are an architecture modeller. Generate a candidate pair of PlantUML views and a machine-readable IR.",
        "The context below is evidence, not permission to invent unsupported technology or deployment facts.",
        "",
        "## Hard constraints",
    ]
    lines.extend(f"- {item}" for item in contract["must_preserve"])
    lines.extend([
        "",
        "## Required output contract",
        "Return JSON with `components`, `relations`, `deployment_environments`, `deployment_nodes`, `deployment_instances`, `deployment_relations`, and `unresolved_concepts`.",
        "Use the pinned restricted PlantUML profile and make component/deployment aliases cross-view consistent.",
        "The `components` array MUST contain exactly one system boundary object with `kind` set to `software_system`, `level` set to `c4_software_system`, `scope` set to `system_of_interest`, and `group` set to true. Every other component must have `group` set to false. Do not omit `group`.",
        "A valid system boundary object has this shape: {\"alias\":\"sys_platform\",\"name\":\"System of interest\",\"kind\":\"software_system\",\"level\":\"c4_software_system\",\"scope\":\"system_of_interest\",\"roles\":[],\"technologies\":[],\"group\":true}.",
        "Use lowercase snake_case aliases, use `sys_platform` for the sole system boundary, and reuse each chosen alias exactly everywhere else.",
        "Do not invent references: every logical relation source/destination must equal a component alias; every deployment instance component_alias must equal a component alias and node_alias must equal a deployment node alias; every deployment relation source_node/destination_node must equal node aliases and logical_relation_id must equal a logical relation_id.",
        "The literal alias `root` is reserved for the parent field of the system boundary; never use `root` as a logical relation endpoint, deployment endpoint, or component alias. If a relation cannot be grounded in two exact component aliases, omit that relation and record the unsupported concept in `unresolved_concepts`.",
        "Before returning JSON, enumerate the exact component aliases mentally and verify that every relation source and destination is one of those aliases. Do not use a label, parent name, system name, or inferred alias as an endpoint.",
        "Every logical relation must include `protocols` as an array plus `mode` and `access` strings; use `unknown` when the evidence does not support a more specific value.",
        "Every component whose kind is not `actor` or `software_system` must appear at least once in `deployment_instances`; include audit/logging components too, even if their hosting technology is unknown.",
        "",
        "## Retrieved evidence and normalized input",
        "```json",
        json.dumps(context, ensure_ascii=False, indent=2),
        "```",
    ])
    return "\n".join(lines) + "\n"


def run_pipeline(
    input_path: Path,
    kb_root: Path,
    output_dir: Path,
    top_k: int = 3,
    accept_mock_review: bool = False,
    generator: str = "mock",
    model: str = "qwen3:4b",
    ollama_url: str = "http://localhost:11434",
    llm_timeout: int = 300,
    semantic_model: str | None = None,
    reranker_model: str | None = None,
    reranker_trust_remote_code: bool = False,
    semantic_trust_remote_code: bool = False,
) -> dict[str, Any]:
    input_data = read_json(input_path)
    requirements = normalize_input(input_data)
    semantic_provider = (
        SentenceTransformerEmbeddingProvider(
            semantic_model,
            trust_remote_code=semantic_trust_remote_code,
        )
        if semantic_model
        else None
    )
    reranker = (
        SentenceTransformerCrossEncoderReranker(
            reranker_model,
            trust_remote_code=reranker_trust_remote_code,
        )
        if reranker_model
        else None
    )
    retrieval = retrieve(
        kb_root,
        requirements,
        top_k=top_k,
        semantic_provider=semantic_provider,
        reranker=reranker,
    )
    context = build_context(kb_root, requirements, retrieval)
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_raw: dict[str, Any] | None = None
    if generator == "ollama":
        ir, llm_raw, llm_metadata = generate_ollama_ir(
            context,
            model=model,
            base_url=ollama_url,
            timeout_seconds=llm_timeout,
        )
    elif generator == "mock":
        ir = generate_mock_ir(context)
        llm_metadata = {
            "model": "deterministic-mock-llm",
            "structured_output": False,
        }
    else:
        raise ValueError(f"Unsupported generator: {generator}")
    if llm_raw is not None:
        write_json(output_dir / "local-llm-response.json", llm_raw)
    _validate_ir_shape(ir)
    component_puml = generate_component_puml(ir)
    deployment_puml = generate_deployment_puml(ir)
    validation = validate_candidate(
        ir,
        component_puml,
        deployment_puml,
        retrieval["retrieval_audit"],
    )
    review = {
        "schema_version": MOCK_SCHEMA_VERSION,
        "candidate_id": ir["candidate_id"],
        "disposition": "accepted_for_demo" if accept_mock_review else "pending_human_review",
        "reviewer": "deterministic-mock-llm" if accept_mock_review else None,
        "feedback_items": [{"severity":"warning", "note":"A real architect must review requirements, evidence, unresolved concepts and both views."}],
        "stage2_eligible_vote": bool(accept_mock_review and validation["overall_status"] == "ready_for_review"),
        "claim_boundary": "accepted_for_demo is not owner acceptance or independent architectural review.",
    }
    manifest = {
        "schema_version": MOCK_SCHEMA_VERSION,
        "status": "local_llm_bundle" if generator == "ollama" else "mock_bundle",
        "candidate_id": ir["candidate_id"],
        "file_count": 0,
        "stage2_eligible": review["stage2_eligible_vote"],
        "review_status": review["disposition"],
        "files": {
            "input": ["input.json", "requirements.json"],
            "evidence": ["retrieval-evidence.json", "architecture-context.json", "llm-prompt.md"],
            "design": ["architecture-ir.json", "component.puml", "deployment.puml", "validation.json"],
            "governance": ["design-review.json", "handoff-gate.json", "manifest.json", "generation-run.json"],
            "stage2": ["experiment-config.json", "stage2-status.json"],
        },
    }
    if generator == "ollama":
        manifest["files"]["llm"] = ["local-llm-response.json"]
    experiment_config = {
        "schema_version": MOCK_SCHEMA_VERSION,
        "candidate_id": ir["candidate_id"],
        "status": "planned_not_executed",
        "architecture_revision": sha256_json(ir),
        "workloads": [{"id":"W-demo", "description":"No workload values are measured by this mock."}],
        "quality_bounds": {},
        "claim_boundary": "Performance context must come from profiling/benchmark execution; no numbers are fabricated here.",
    }
    gate_checks = {
        "owner_acceptance": review["disposition"] == "accepted_for_demo",
        "stage2_eligible": review["stage2_eligible_vote"],
        "experiment_config_present": True,
        "stage2_adapters_ready": False,
    }
    gate = {
        "schema_version": MOCK_SCHEMA_VERSION,
        "candidate_id": ir["candidate_id"],
        "gate_result": "eligible_for_demo_only" if all(gate_checks.values()) else "blocked",
        "checks": gate_checks,
        "blocked_reasons": [key for key, value in gate_checks.items() if not value],
        "claim_boundary": "This prototype does not implement prototype or PCM adapters.",
    }
    stage2_status = {
        "prototype": {"status":"not_implemented", "artifacts":[]},
        "simulation": {"status":"not_implemented", "artifacts":[]},
        "comparison": {"status":"not_implemented", "automatic_winner":None},
    }
    write_json(output_dir / "input.json", input_data)
    write_json(output_dir / "requirements.json", requirements)
    write_json(output_dir / "retrieval-evidence.json", retrieval)
    write_json(output_dir / "architecture-context.json", context)
    write_json(output_dir / "architecture-ir.json", ir)
    write_json(output_dir / "validation.json", validation)
    write_json(output_dir / "design-review.json", review)
    write_json(output_dir / "experiment-config.json", experiment_config)
    write_json(output_dir / "handoff-gate.json", gate)
    write_json(output_dir / "stage2-status.json", stage2_status)
    (output_dir / "component.puml").write_text(component_puml, encoding="utf-8")
    (output_dir / "deployment.puml").write_text(deployment_puml, encoding="utf-8")
    (output_dir / "llm-prompt.md").write_text(generate_prompt(context), encoding="utf-8")
    generation_run = {
        "schema_version": MOCK_SCHEMA_VERSION,
        "run_id": f"run-{sha256_json({'input': input_data, 'kb': str(kb_root)})[:12]}",
        "created_at": utc_now(),
        "model": llm_metadata["model"],
        "generator": generator,
        "llm": llm_metadata,
        "retrieval_index": retrieval["index_id"],
        "profile_version": PROFILE_VERSION,
        "output_hashes": {
            path.name: sha256_bytes(path.read_bytes())
            for path in output_dir.iterdir()
            if path.is_file() and path.name not in {"manifest.json", "generation-run.json"}
        },
        "claim_boundary": "For mock runs this is deterministic generation; for Ollama runs it records a local LLM generation. Neither is production architecture proof.",
    }
    write_json(output_dir / "generation-run.json", generation_run)
    manifest["file_count"] = len([
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    ]) + 1
    write_json(output_dir / "manifest.json", manifest)
    return {"requirements": requirements, "retrieval": retrieval, "ir": ir, "validation": validation, "gate": gate, "output_dir": str(output_dir)}
