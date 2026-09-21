import unittest

from arch_context_pipeline.adaptive_retrieval import (
    AdaptiveBanditConfig,
    select_adaptive_evidence,
)


class AdaptiveRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retrieval = {
            "queries": [
                {
                    "query_id": "Q-A",
                    "evidence": [
                        {"rank": 1, "evidence_id": "d-a1", "text": "A1"},
                        {"rank": 2, "evidence_id": "d-a2", "text": "A2"},
                    ],
                },
                {
                    "query_id": "Q-B",
                    "evidence": [
                        {"rank": 1, "evidence_id": "d-b1", "text": "B1"},
                        {"rank": 2, "evidence_id": "d-b2", "text": "B2"},
                    ],
                },
            ]
        }

    def test_warm_start_and_budget_are_reproducible(self):
        labels = {"Q-A": {"d-a1": 1}, "Q-B": {"d-b1": 0}}
        config = AdaptiveBanditConfig(budget=4, seed=7, warm_start=True)
        first = select_adaptive_evidence(self.retrieval, labels, config=config)
        second = select_adaptive_evidence(self.retrieval, labels, config=config)
        self.assertEqual(first["trace"], second["trace"])
        self.assertEqual(first["selected_pull_count"], 4)
        self.assertEqual(first["trace"][0]["decision"], "warm_start")
        self.assertEqual(first["trace"][1]["decision"], "warm_start")

    def test_beta_posterior_updates_from_binary_reward(self):
        labels = {"Q-A": {"d-a1": 1}, "Q-B": {"d-b1": 0}}
        result = select_adaptive_evidence(
            self.retrieval,
            labels,
            config=AdaptiveBanditConfig(budget=2, seed=1, warm_start=True),
        )
        self.assertEqual(result["final_posteriors"]["Q-A"]["alpha"], 2.0)
        self.assertEqual(result["final_posteriors"]["Q-A"]["beta"], 1.0)
        self.assertEqual(result["final_posteriors"]["Q-B"]["alpha"], 1.0)
        self.assertEqual(result["final_posteriors"]["Q-B"]["beta"], 2.0)

    def test_missing_label_can_be_rejected(self):
        with self.assertRaises(ValueError):
            select_adaptive_evidence(
                self.retrieval,
                {},
                config=AdaptiveBanditConfig(budget=1, unjudged_policy="error"),
            )

    def test_top_k_ucb_policy_is_available(self):
        labels = {"Q-A": {"d-a1": 1}, "Q-B": {"d-b1": 0}}
        result = select_adaptive_evidence(
            self.retrieval,
            labels,
            config=AdaptiveBanditConfig(budget=2, policy="top_k_ucb", warm_start=False),
        )
        self.assertEqual(result["method"]["name"], "bernoulli_top_k_ucb_research_adapter")
        self.assertFalse(result["method"]["diversity_applied"])


if __name__ == "__main__":
    unittest.main()
