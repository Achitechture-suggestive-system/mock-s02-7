import json
import math
import tempfile
import unittest
from pathlib import Path

from arch_context_pipeline.pipeline import (
    RETRIEVAL_FIELDS,
    RetrievalConfig,
    _bm25f_score,
    build_evidence_units,
    build_context,
    generate_component_puml,
    generate_deployment_puml,
    generate_mock_ir,
    local_llm_ir_schema,
    normalize_input,
    retrieve,
    reciprocal_rank_fusion,
    run_pipeline,
    validate_candidate,
)


KB = Path(r"C:\disk D\KnowledgeBase_SoftwareArchitect")
EXAMPLE = Path(__file__).parents[1] / "examples" / "clinic_input.json"
OOD_EXAMPLE = Path(__file__).parents[1] / "examples" / "out_of_domain_cryobot_input.json"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        cls.requirements = normalize_input(cls.input_data)
        cls.retrieval = retrieve(KB, cls.requirements, top_k=3)
        cls.context = build_context(KB, cls.requirements, cls.retrieval)
        cls.ir = generate_mock_ir(cls.context)
        cls.component = generate_component_puml(cls.ir)
        cls.deployment = generate_deployment_puml(cls.ir)

    def test_retrieval_has_evidence_and_scores(self):
        self.assertEqual(self.retrieval["method"]["lexical"]["name"], "bm25f")
        self.assertGreaterEqual(len(self.retrieval["queries"]), 4)
        self.assertTrue(all(len(query["evidence"]) == 3 for query in self.retrieval["queries"]))
        self.assertIn("evidence_id", self.retrieval["queries"][0]["evidence"][0])
        self.assertIn("source_locator", self.retrieval["queries"][0]["evidence"][0])
        self.assertEqual(self.retrieval["method"]["semantic"]["status"], "not_configured")
        self.assertIn("retrieval_audit", self.retrieval)
        self.assertIn("matched_content_terms", self.retrieval["queries"][0]["evidence"][0])

    def test_normalize_preserves_raw_text_for_traceability(self):
        self.assertEqual(self.requirements["raw_text"], self.input_data["raw_text"])

    def test_manifest_file_count_matches_rerun_bundle(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "bundle"
            run_pipeline(EXAMPLE, KB, output_dir, top_k=1)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["file_count"], len(list(output_dir.iterdir())))

    def test_out_of_domain_input_is_flagged_by_retrieval_audit(self):
        data = json.loads(OOD_EXAMPLE.read_text(encoding="utf-8"))
        result = retrieve(KB, normalize_input(data), top_k=3)

        audit = result["retrieval_audit"]
        self.assertEqual(audit["overall_status"], "out_of_domain_candidate")
        self.assertEqual(len(audit["weak_query_ids"]), len(result["queries"]))
        self.assertTrue(any(item["oov_content_terms"] for item in audit["queries"]))
        self.assertTrue(all(query["support_status"] != "lexically_supported" for query in result["queries"]))

        context = build_context(KB, normalize_input(data), result)
        ir = generate_mock_ir(context)
        component = generate_component_puml(ir)
        deployment = generate_deployment_puml(ir)
        validation = validate_candidate(ir, component, deployment, audit)
        self.assertEqual(validation["overall_status"], "invalid")
        self.assertEqual(validation["retrieval_audit_status"], "out_of_domain_candidate")
        self.assertEqual(next(check for check in validation["checks"] if check["check_id"] == "CHK-008")["result"], "fail")

    def test_lexical_exactness(self):
        data = {
            "case_id": "case-test-exact",
            "raw_text": "A healthcare interface shall support FHIR R4 and SMART-on-FHIR profiles.",
            "functional_requirements": [{
                "id": "FR-EXACT",
                "statement": "The healthcare interface shall support FHIR R4 and SMART-on-FHIR profiles.",
            }],
        }
        result = retrieve(KB, normalize_input(data), top_k=1)
        exact_query = next(query for query in result["queries"] if query["requirement_id"] == "FR-EXACT")
        self.assertEqual(exact_query["evidence"][0]["case_id"], "case-000008")
        self.assertIn("FHIR", exact_query["evidence"][0]["text"])

    def test_semantic_vietnamese_english_fixture(self):
        class FixtureEmbedding:
            model_name = "fixture-cross-lingual"
            vectors_are_l2_normalized = True

            def encode(self, texts):
                vectors = []
                for text in texts:
                    lowered = text.casefold()
                    healthcare_access = any(term in lowered for term in (
                        "thông tin y tế", "ứng dụng được cấp quyền", "healthcare information", "fhir",
                    ))
                    vectors.append([1.0, 0.0] if healthcare_access else [0.0, 1.0])
                return vectors

        data = {
            "case_id": "case-test-vi",
            "raw_text": "Ứng dụng được cấp quyền phải nhận thông tin y tế.",
            "functional_requirements": [{
                "id": "FR-VI",
                "statement": "Ứng dụng được cấp quyền phải nhận thông tin y tế.",
            }],
        }
        result = retrieve(KB, normalize_input(data), top_k=10, semantic_provider=FixtureEmbedding())
        query = next(item for item in result["queries"] if item["requirement_id"] == "FR-VI")
        healthcare = next(item for item in query["evidence"] if item["case_id"] == "case-000008")
        self.assertEqual(healthcare["semantic_rank"], 1)
        self.assertIsNotNone(healthcare["semantic_score"])

    def test_cross_view_alias_consistency(self):
        aliases = {element["alias"] for element in self.ir["components"]}
        for instance in self.ir["deployment_instances"]:
            self.assertIn(instance["component_alias"], aliases)
            self.assertIn(f"instance-of:{instance['component_alias']}", self.deployment)
        self.assertTrue(any(element["kind"] == "software_system" for element in self.ir["components"]))
        self.assertTrue(any(element["kind"] == "deployable_unit" for element in self.ir["components"]))

    def test_english_example_aligns_with_openemr_evidence(self):
        context_query = next(query for query in self.retrieval["queries"] if query["requirement_id"] == "CONTEXT-001")
        openemr_evidence = next(item for item in context_query["evidence"] if item["case_id"] == "case-000008")
        self.assertGreaterEqual(len(openemr_evidence["matched_terms"]), 8)

        fhir_query = next(query for query in self.retrieval["queries"] if query["requirement_id"] == "NFR-INT-001")
        self.assertEqual(fhir_query["evidence"][0]["case_id"], "case-000008")
        self.assertGreaterEqual(len(fhir_query["evidence"][0]["matched_terms"]), 5)

    def test_static_validation_is_ready_for_review(self):
        result = validate_candidate(self.ir, self.component, self.deployment)
        self.assertEqual(result["overall_status"], "ready_for_review")
        self.assertTrue(all(check["result"] != "fail" for check in result["checks"]))

    def test_no_fake_performance_values(self):
        self.assertEqual(self.context["generation_contract"]["human_review_required"], True)
        self.assertNotIn("p95_ms", json.dumps(self.ir))

    def test_kb_integrity_is_visible_in_context(self):
        preflight = self.context["profile"]["integrity_preflight"]
        self.assertIn(preflight["status"], {"pass", "warning"})
        self.assertIn("checks", preflight)

    def test_local_llm_schema_requires_cross_view_fields(self):
        schema = local_llm_ir_schema()
        component_required = schema["properties"]["components"]["items"]["required"]
        relation_required = schema["properties"]["relations"]["items"]["required"]
        self.assertIn("group", component_required)
        self.assertIn("protocols", relation_required)
        self.assertIn("mode", relation_required)
        self.assertIn("access", relation_required)

    def test_rrf_matches_manual_calculation(self):
        fused = reciprocal_rank_fusion({
            "A": ["d1", "d2"],
            "B": ["d2", "d1"],
        }, k=60)
        scores = {item["item_id"]: item["rrf_score"] for item in fused}
        self.assertAlmostEqual(scores["d1"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(scores["d2"], 1 / 62 + 1 / 61)
        self.assertEqual([item["item_id"] for item in fused], ["d1", "d2"])

    def test_bm25f_matches_manual_calculation(self):
        config = RetrievalConfig(
            k1=1.2,
            field_weights={field_name: 1.0 for field_name in RETRIEVAL_FIELDS},
            field_b={field_name: 0.0 for field_name in RETRIEVAL_FIELDS},
        )
        score = _bm25f_score(
            {"alpha": 1},
            {"problem_statement": {"alpha": 2}},
            {"problem_statement": 2.0},
            {"alpha": math.log(5 / 3)},
            config,
        )
        expected = ((1.2 + 1) * 2 / (1.2 + 2)) * math.log(5 / 3)
        self.assertAlmostEqual(score, expected)

    def test_provenance_and_stable_evidence_ids(self):
        first_units, first_exclusions = build_evidence_units(
            KB, json.loads((KB / "derived" / "retrieval" / "production-index.json").read_text(encoding="utf-8"))
        )
        second_units, second_exclusions = build_evidence_units(
            KB, json.loads((KB / "derived" / "retrieval" / "production-index.json").read_text(encoding="utf-8"))
        )
        self.assertEqual(
            [(unit["evidence_id"], unit["text"]) for unit in first_units],
            [(unit["evidence_id"], unit["text"]) for unit in second_units],
        )
        self.assertEqual(first_exclusions, second_exclusions)
        for unit in first_units:
            self.assertTrue((KB / unit["source_path"]).exists())

    def test_no_leakage_and_architecture_hint_isolation(self):
        data = {
            "case_id": "case-test-leakage",
            "raw_text": "The service shall expose healthcare information.",
            "functional_requirements": [{
                "id": "FR-KEEP",
                "statement": "The service shall expose healthcare information.",
            }],
            "constraints": [{
                "id": "C-LEAK",
                "statement": "Use the reference deployment exactly.",
                "rag_eligible": False,
                "leakage": {"classification": "solution_leakage", "action": "exclude_from_rag"},
            }],
            "architecture_hints": {"database": "database.postgresql"},
        }
        changed_hints = dict(data)
        changed_hints["architecture_hints"] = {"database": "database.sqlite", "cache": "cache.redis"}
        left = retrieve(KB, normalize_input(data), top_k=1)
        right = retrieve(KB, normalize_input(changed_hints), top_k=1)
        self.assertEqual(
            [query["query_text"] for query in left["queries"]],
            [query["query_text"] for query in right["queries"]],
        )
        self.assertEqual(
            [[item["evidence_id"] for item in query["evidence"]] for query in left["queries"]],
            [[item["evidence_id"] for item in query["evidence"]] for query in right["queries"]],
        )
        self.assertIn("C-LEAK", json.dumps(left["exclusions"]))


if __name__ == "__main__":
    unittest.main()
