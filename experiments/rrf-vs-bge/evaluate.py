"""Evaluate RRF versus BGE cross-encoder reranking on a labelled KB fixture.

The experiment keeps the lexical retriever, dense embedding model, RRF k,
candidate pool, and top-k fixed.  The only changed stage is the final ranking:

* RRF: lexical + dense rank fusion is returned directly.
* BGE: the same top-M RRF candidates are rescored by the cross-encoder.

This is a retrieval/reranking benchmark only; it intentionally does not call
Ollama or the architecture generator.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from arch_context_pipeline.pipeline import (
    RetrievalConfig,
    SentenceTransformerCrossEncoderReranker,
    SentenceTransformerEmbeddingProvider,
    normalize_input,
    retrieve,
    write_json,
)


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BGE_MODEL = "BAAI/bge-reranker-v2-m3"


# Each pair targets one real evidence unit. The first query preserves much of
# the source vocabulary; the second is deliberately paraphrased to test the
# ranking stage beyond simple exact-token overlap.
CASES: list[tuple[str, str, str, str, str]] = [
    ("case-000006:E003", "commerce", "The platform must expose commerce capabilities to storefronts and admin clients through network APIs.", "Commerce clients need a network API boundary for storefront and back-office functionality.",),
    ("case-000006:E006", "commerce", "Third-party commerce services must be replaceable without rebuilding the whole platform.", "Swap external commerce infrastructure independently while keeping the main platform intact.",),
    ("case-000007:E003", "crm", "Users must be able to manage CRM objects, views, and workflows.", "CRM users need to maintain business records, saved views, and process flows.",),
    ("case-000007:E005", "crm", "Asynchronous CRM work must run outside the request-serving process.", "Long-running CRM operations should not block interactive web requests.",),
    ("case-000008:E004", "healthcare", "Healthcare information must be exposed to authorized applications through REST and FHIR interfaces.", "External clinical applications need authorized REST and FHIR access to health data.",),
    ("case-000008:E006", "healthcare", "The healthcare interface must support FHIR R4 and SMART-on-FHIR profiles.", "Clinical interoperability requires the FHIR version four and SMART on FHIR standards.",),
    ("case-000009:E003", "offline-learning", "The platform must provide teaching and learning functions without an Internet connection.", "Learners and teachers need core education features while offline.",),
    ("case-000009:E005", "offline-learning", "Partitioned facility data must synchronize with peer instances or the data portal.", "Disconnected facilities need selected records synchronized with peers or the central portal.",),
    ("case-000010:E005", "digital-twin", "The system must maintain searchable twin state and exchange messages with external systems.", "Digital-twin state should be searchable and able to communicate with outside systems.",),
    ("case-000010:E006", "digital-twin", "Applications must interact with twins without depending on the device connection protocol.", "Clients should use digital twins independently of each physical device protocol.",),
    ("case-000011:E003", "iot-cloud", "Device telemetry and events must be ingested through protocol adapters.", "The cloud backend needs adapters that collect readings and events from device protocols.",),
    ("case-000011:E005", "iot-cloud", "Application commands must be routed to the adapter connected to the target device.", "Commands from applications need delivery to the protocol adapter for the chosen device.",),
    ("case-000012:E003", "sparql", "RDF datasets must be exposed through a network-accessible SPARQL service.", "Clients need a remote semantic-web endpoint for querying RDF data with SPARQL.",),
    ("case-000012:E006", "sparql", "Persistent datasets must survive server-container termination when mounted outside it.", "RDF data stored on an external mount must remain after the service container stops.",),
    ("case-000013:E003", "workflow", "Dependent work must be represented as directed acyclic workflows and tasks.", "The scheduler needs a DAG model for tasks with dependencies.",),
    ("case-000013:E005", "workflow", "Operators must inspect, trigger, and debug workflows through an API and user interface.", "Workflow owners need API and UI controls to examine, start, and troubleshoot runs.",),
    ("case-000014:E004", "federated-messaging", "The system must exchange and reconcile events with other homeservers.", "Messaging servers need federation that synchronizes and resolves events with peer servers.",),
    ("case-000014:E006", "federated-messaging", "Large installations must scale worker processes horizontally and independently.", "High-volume deployments should add worker replicas without scaling every other service.",),
    ("case-000015:E004", "osm", "XML and JSON interfaces must be exposed for editing OpenStreetMap data.", "Map editors need XML and JSON APIs for changing OpenStreetMap records.",),
    ("case-000015:E005", "osm", "The system must accept and expose GPX trace uploads.", "Users need to upload and retrieve GPS exchange traces.",),
    ("case-000016:E003", "crisis-mapping", "Information must be collected from web, SMS, email, feeds, and social channels.", "The reporting platform needs ingestion across browser, text message, email, feed, and social sources.",),
    ("case-000016:E004", "crisis-mapping", "Collected information must be transformed, categorized, and geolocated as posts.", "Incoming reports need normalization, classification, and geographic positioning.",),
    ("case-000017:E003", "billing", "Usage events must be converted into billable metrics and charges.", "The billing system needs to turn product usage into metered revenue amounts.",),
    ("case-000017:E004", "billing", "The platform must generate invoices and coordinate payment collection.", "Billing must create customer invoices and manage collecting the money owed.",),
    ("case-000018:E007", "commerce-jobs", "Non-request work such as email, image processing, webhooks, and catalog import must run as background jobs.", "Email, media processing, webhook delivery, and imports should execute asynchronously away from requests.",),
    ("case-000018:E008", "commerce-jobs", "The platform must support caching expensive computations and an in-memory cache for high-traffic installations.", "High-traffic commerce deployments need memory caching for costly calculations.",),
    ("case-000019:E006", "low-code-crm", "The platform must support flexible role-based security and configurable privacy controls.", "Business applications need customizable RBAC and privacy policy enforcement.",),
    ("case-000019:E007", "low-code-crm", "The platform must remain integrable with external services and other instances through its API-centric design.", "An API-first business platform must connect to outside services and peer installations.",),
    ("case-000020:E003", "public-health", "Program data must be captured, managed, and validated from browser, Android, feature-phone, and SMS clients.", "Public-health data collection must work across web, Android, basic phones, and text messages.",),
    ("case-000020:E004", "public-health", "The system must provide dashboards, pivots, charts, GIS, and other analytics visualizations.", "Health-program users need analytical dashboards, pivot views, charts, and maps.",),
    ("case-000021:E004", "learning-platform", "Educators, administrators, and learners must create and use personalized learning environments.", "The education platform needs personalized spaces for teachers, managers, and students.",),
    ("case-000021:E008", "learning-platform", "The platform must expose one unified management interface for standard and extension plugin types.", "Administrators need a common control surface for built-in and add-on plugins.",),
    ("case-000022:E004", "federated-social", "The system must interoperate with independently operated ActivityPub servers.", "The social network must federate with separately administered ActivityPub instances.",),
    ("case-000022:E006", "federated-social", "Published statuses must be distributed asynchronously to recipient timelines.", "New social posts should reach followers' timelines through background delivery.",),
    ("case-000023:E003", "subscription-billing", "The system must manage recurring subscriptions and their billing.", "The platform needs lifecycle management for subscriptions and periodic charges.",),
    ("case-000023:E006", "subscription-billing", "Billing changes must be coordinated over notification and persistent event queues.", "Subscription updates need durable event and notification queues for coordination.",),
    ("case-000024:E004", "civil-registration", "The event service must manage custom events and expose tRPC plus selected REST endpoints for country integrations.", "Civil-registration country systems need event operations through tRPC and some REST APIs.",),
    ("case-000024:E005", "civil-registration", "The gateway must route client requests to authentication, event, and document services.", "A gateway should dispatch incoming requests to identity, event, and document backends.",),
    ("case-000025:E004", "iot-platform", "Devices must connect through protocol agents and manager APIs including MQTT, HTTP/REST, and WebSocket.", "The IoT platform needs MQTT, REST, and WebSocket connectivity through agents and managers.",),
    ("case-000025:E005", "iot-platform", "Asset behavior must be automated through when-then, flow, and script rules.", "Device and asset automation should be driven by conditional, flow-based, and scripted rules.",),
    ("case-000026:E003", "metadata", "Metadata must be ingested from external systems through pull, push, synchronous, and asynchronous mechanisms.", "The catalog needs multiple ingestion modes for metadata sources, including pull and push.",),
    ("case-000026:E005", "metadata", "Committed metadata changes must update search and graph projections from the change log.", "A metadata change log should drive refreshed search and graph views.",),
    ("case-000027:E003", "geospatial", "Users must upload, publish, discover, and visualize geospatial data and metadata.", "Non-specialists need to import, find, publish, and map geographic datasets.",),
    ("case-000027:E009", "geospatial", "Authentication and authorization must be synchronized across GeoNode and pluggable geospatial services using OAuth2.", "Geospatial services need shared OAuth2 identity and access decisions with the main platform.",),
    ("case-000028:E005", "commerce-modular", "Catalog, cart, order, and checkout behavior must execute through independent commerce modules.", "Commerce capabilities should be split into independently replaceable catalog, cart, order, and checkout modules.",),
    ("case-000028:E010", "commerce-modular", "Remote app backends must communicate through stable API and webhook contracts.", "Externally hosted extensions need durable API and webhook integration contracts.",),
    ("case-000029:E005", "crm-extensions", "Backend, module, and frontend extensions must be packaged independently.", "CRM customizations need independently deployable server, module, and UI packages.",),
    ("case-000029:E006", "crm-extensions", "Scheduled workflows and asynchronous tasks must run outside interactive requests.", "CRM jobs triggered by schedules should execute in background workers, not web requests.",),
    ("case-000030:E006", "edge-iot", "Messages must be transformed, filtered, and routed through configurable processing pipelines.", "Edge data needs configurable pipelines for conversion, filtering, and routing.",),
    ("case-000030:E010", "edge-iot", "Store-and-forward behavior must be retained for intermittent connectivity.", "The edge system must buffer data and forward it later when connectivity is unreliable.",),
    ("case-000031:E003", "aerial-imagery", "Aerial image sets must produce georeferenced maps, point clouds, elevation models, and 3D outputs.", "The processing service should turn drone imagery into maps, point clouds, terrain models, and 3D products.",),
    ("case-000031:E005", "aerial-imagery", "The system must select an online processing node with the lowest queue when automatic selection is enabled.", "Automatic job placement should choose the available processing worker with the shortest queue.",),
    ("case-000032:E004", "scientific-workflows", "Submitted jobs must be assigned to dedicated handlers and configurable execution destinations.", "Scientific jobs need routing to specialized handlers and selectable compute backends.",),
    ("case-000032:E009", "scientific-workflows", "Cluster-submitted jobs must remain trackable across a Galaxy server restart.", "Remote scientific jobs must keep their tracking state after the web server restarts.",),
    ("case-000033:E004", "research-repository", "The repository must support sharing, finding, citing, and preserving research datasets.", "Researchers need dataset discovery, sharing, citation, and long-term preservation.",),
    ("case-000033:E007", "research-repository", "Tabular ingestion and indexing work must execute asynchronously.", "Large table imports and search indexing should be handled by background workers.",),
    ("case-000034:E003", "file-collaboration", "Users must store, synchronize, and share files and collaboration data.", "A self-hosted collaboration service needs file storage, sync, and sharing.",),
    ("case-000034:E008", "file-collaboration", "Clustered deployments must share one distributed cache and support multiple application replicas.", "Multiple application instances need a common distributed cache in a cluster.",),
    ("case-000035:E004", "team-messaging", "The platform must expose REST APIs and persistent WebSocket connections to supported clients.", "Messaging clients need both HTTP APIs and long-lived WebSocket sessions.",),
    ("case-000035:E007", "team-messaging", "The deployment must maintain service through redundant application, database, and load-balancing infrastructure.", "Reliable team messaging requires redundant app, database, and load-balancer components.",),
]


class CachedEmbeddingProvider:
    def __init__(self, base: SentenceTransformerEmbeddingProvider) -> None:
        self.base = base
        self.model_name = base.model_name
        self.vectors_are_l2_normalized = base.vectors_are_l2_normalized
        self.trust_remote_code = getattr(base, "trust_remote_code", False)
        self.cache: dict[tuple[str, ...], list[list[float]]] = {}

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        key = tuple(str(text) for text in texts)
        if key not in self.cache:
            self.cache[key] = self.base.encode(key)
        return self.cache[key]


def make_input(query_id: str, query: str) -> dict[str, Any]:
    return normalize_input({
        "case_id": f"eval-{query_id}",
        "canonical_language": "en",
        "raw_text": "Retrieval ranking benchmark input.",
        "problem_statement": "Retrieval ranking benchmark input.",
        "system_summary": "Retrieval ranking benchmark input.",
        "functional_requirements": [{
            "id": query_id,
            "statement": query,
            "priority": "must",
            "source_refs": [{"source_id": "SRC-EVAL", "locator": "benchmark.query"}],
            "rag_eligible": True,
            "leakage": {"classification": "none", "action": "keep"},
        }],
        "actors": [],
    })


def rank_of(rows: Sequence[dict[str, Any]], target: str, field: str) -> int | None:
    for row in rows:
        if row.get("evidence_id") == target:
            value = row.get(field)
            return int(value) if value is not None else None
    return None


def reciprocal(rank: int | None, cutoff: int) -> float:
    return 1.0 / rank if rank is not None and rank <= cutoff else 0.0


def summarize(results: list[dict[str, Any]], name: str, rank_key: str) -> dict[str, Any]:
    cutoffs = (1, 3, 5, 10, 50)
    return {
        "name": name,
        "queries": len(results),
        "candidate_recall_at_50": round(sum(item["candidate_rank"] is not None for item in results) / len(results), 6),
        "hit_rate": {f"@{cutoff}": round(sum(item[rank_key] is not None and item[rank_key] <= cutoff for item in results) / len(results), 6) for cutoff in cutoffs},
        "mrr": {f"@{cutoff}": round(sum(reciprocal(item[rank_key], cutoff) for item in results) / len(results), 6) for cutoff in cutoffs},
        "mean_rank_when_found_at_50": round(statistics.mean([item[rank_key] for item in results if item[rank_key] is not None]), 6),
        "median_rank_when_found_at_50": statistics.median([item[rank_key] for item in results if item[rank_key] is not None]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-root", type=Path, default=Path(r"C:\disk D\KnowledgeBase_SoftwareArchitect"))
    parser.add_argument("--output", type=Path, default=Path("experiments/rrf-vs-bge/benchmark-result.json"))
    args = parser.parse_args()

    if len(CASES) != 60:
        raise RuntimeError(f"Expected 60 benchmark inputs, got {len(CASES)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}", flush=True)
    embedding = CachedEmbeddingProvider(SentenceTransformerEmbeddingProvider(EMBEDDING_MODEL, batch_size=32))
    print(f"Loading reranker model: {BGE_MODEL}", flush=True)
    bge = SentenceTransformerCrossEncoderReranker(BGE_MODEL, batch_size=16)
    config = RetrievalConfig(rrf_k=60, candidate_multiplier=5)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, (target, domain, source_query, paraphrase) in enumerate(CASES, 1):
        for variant, query_text in (("source_vocab", source_query), ("paraphrase", paraphrase)):
            query_id = f"EVAL-{index:03d}-{variant}"
            normalized = make_input(query_id, query_text)
            rrf_output = retrieve(args.kb_root, normalized, top_k=10, config=config, semantic_provider=embedding)
            bge_output = retrieve(args.kb_root, normalized, top_k=10, config=config, semantic_provider=embedding, reranker=bge)
            rrf_query = next(item for item in rrf_output["queries"] if item["query_id"] == query_id)
            bge_query = next(item for item in bge_output["queries"] if item["query_id"] == query_id)
            rrf_rank = rank_of(rrf_query["candidate_pool"], target, "rrf_rank")
            bge_rank = rank_of(bge_query["candidate_pool"], target, "reranker_rank")
            candidate_rank = rank_of(rrf_query["candidate_pool"], target, "rrf_rank")
            rrf_top = [row["evidence_id"] for row in rrf_query["evidence"]]
            bge_top = [row["evidence_id"] for row in bge_query["evidence"]]
            rows.append({
                "input_id": query_id,
                "domain": domain,
                "variant": variant,
                "query": query_text,
                "gold_evidence_id": target,
                "candidate_rank": candidate_rank,
                "rrf_rank": rrf_rank,
                "bge_rank": bge_rank,
                "rrf_top10": rrf_top,
                "bge_top10": bge_top,
                "rrf_top1": rrf_top[0] if rrf_top else None,
                "bge_top1": bge_top[0] if bge_top else None,
                "bge_rank_delta_vs_rrf": (bge_rank - rrf_rank) if bge_rank is not None and rrf_rank is not None else None,
                "rrf_support_status": rrf_query["support_status"],
                "bge_support_status": bge_query["support_status"],
                "rrf_target_row": next((row for row in rrf_query["candidate_pool"] if row["evidence_id"] == target), None),
                "bge_target_row": next((row for row in bge_query["candidate_pool"] if row["evidence_id"] == target), None),
            })
        print(f"[{index:02d}/{len(CASES)}] completed {domain}; elapsed={time.perf_counter() - started:.1f}s", flush=True)

    comparable = [row for row in rows if row["rrf_rank"] is not None and row["bge_rank"] is not None]
    improved = [row for row in comparable if row["bge_rank"] < row["rrf_rank"]]
    harmed = [row for row in comparable if row["bge_rank"] > row["rrf_rank"]]
    unchanged = [row for row in comparable if row["bge_rank"] == row["rrf_rank"]]
    top1_agreement = sum(row["rrf_top1"] == row["bge_top1"] for row in rows) / len(rows)
    bge_top3_overlap = sum(len(set(row["rrf_top10"][:3]) & set(row["bge_top10"][:3])) / 3 for row in rows) / len(rows)

    output = {
        "benchmark": {
            "name": "RRF versus BGE-reranker-v2-m3 labelled reranking evaluation",
            "scope": "retrieval and reranking only; no Ollama generation",
            "inputs": len(rows),
            "gold_labels": "one target evidence unit per input, sourced from the production KB",
            "candidate_pool_size": 50,
            "top_k": 10,
            "rrf_k": 60,
            "shared_embedding_model": EMBEDDING_MODEL,
            "rrf_definition": "lexical BM25F rank + dense embedding rank fused by reciprocal rank fusion",
            "bge_definition": "same top-50 RRF candidates scored by a pairwise cross-encoder; final order by BGE score",
            "bge_model": BGE_MODEL,
        },
        "aggregate": {
            "rrf": summarize(rows, "semantic + lexical RRF", "rrf_rank"),
            "bge": summarize(rows, "semantic + lexical RRF + BGE cross-encoder", "bge_rank"),
            "candidate_pool_recall_at_50": round(sum(row["candidate_rank"] is not None for row in rows) / len(rows), 6),
            "top1_agreement": round(top1_agreement, 6),
            "mean_top3_overlap": round(bge_top3_overlap, 6),
            "rank_comparison": {
                "comparable": len(comparable),
                "bge_improved": len(improved),
                "bge_harmed": len(harmed),
                "unchanged": len(unchanged),
                "mean_delta_bge_minus_rrf": round(statistics.mean([row["bge_rank_delta_vs_rrf"] for row in comparable]), 6),
                "median_delta_bge_minus_rrf": statistics.median([row["bge_rank_delta_vs_rrf"] for row in comparable]),
            },
        },
        "interpretation": {
            "positive_delta": "BGE placed the gold evidence lower than RRF; negative is an improvement.",
            "important_boundary": "This measures retrieval rank against the supplied gold labels, not architecture quality or generation correctness.",
            "candidate_recall_boundary": "If the target is absent from top-50 RRF candidates, BGE cannot recover it.",
            "rrf_is_not_a_model": "RRF is a rank-fusion algorithm; BGE is a learned cross-encoder reranker.",
        },
        "rows": rows,
    }
    write_json(args.output, output)
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
