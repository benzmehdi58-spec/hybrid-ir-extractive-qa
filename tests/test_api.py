"""Automated tests for the NLP Pipeline FastAPI backend.

Runs against the loaded models and artifacts using FastAPI's TestClient.
"""

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import backend.main as backend_main


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(backend_main.app)

    def test_root_and_health(self):
        root_res = self.client.get("/")
        self.assertEqual(root_res.status_code, 200)
        self.assertEqual(root_res.json()["status"], "ok")

        health_res = self.client.get("/health")
        self.assertEqual(health_res.status_code, 200)
        data = health_res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["corpus_size"], 20233)
        self.assertIn("method", data["summ_config"])

    def test_preprocess_preserves_numbers(self):
        tokens = backend_main.tokenize_simple("When did World War II end in 1945?")
        self.assertIn("1945", tokens)
        self.assertIn("world", tokens)

    def test_retrieve_valid_and_invalid(self):
        res = self.client.post(
            "/api/retrieve",
            json={"query": "Who invented the telephone?", "top_k": 3, "alpha": 0.5},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(len(payload["results"]), 3)
        for item in payload["results"]:
            self.assertGreaterEqual(item["tfidf_score"], 0.0)
            self.assertLessEqual(item["tfidf_score"], 1.0001)
            self.assertGreaterEqual(item["bm25_score"], 0.0)
            self.assertLessEqual(item["bm25_score"], 1.0001)

        bad = self.client.post("/api/retrieve", json={"query": "   ", "top_k": 3})
        self.assertEqual(bad.status_code, 400)

        bad_k = self.client.post("/api/retrieve", json={"query": "test", "top_k": 0})
        self.assertEqual(bad_k.status_code, 422)

    def test_summarize_single_and_multi_sentence(self):
        passage = (
            "Albert Einstein developed the theory of relativity in the early twentieth century. "
            "Special relativity was published in 1905 and reshaped modern physics. "
            "General relativity followed in 1915, describing gravity as spacetime curvature. "
            "His photoelectric effect work earned him the 1921 Nobel Prize in Physics. "
            "Relativity remains foundational to astrophysics and GPS navigation today."
        )
        res = self.client.post("/api/summarize", json={"passage": passage})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 5)
        self.assertGreaterEqual(data["kept"], 1)
        self.assertTrue(len(data["summary"]) > 0)

        empty = self.client.post("/api/summarize", json={"passage": "   "})
        self.assertEqual(empty.status_code, 400)

    def test_distilbert_qa_answerable_and_unanswerable(self):
        ans_res = self.client.post(
            "/api/qa/distilbert",
            json={
                "question": "Who invented the telephone?",
                "context": "Alexander Graham Bell was a Scottish-born inventor credited with patenting the first practical telephone in 1876.",
            },
        )
        self.assertEqual(ans_res.status_code, 200)
        ans_data = ans_res.json()
        self.assertTrue(ans_data["answerable"])
        self.assertIn("bell", ans_data["answer"].lower())
        self.assertLess(ans_data["start_token"], len(ans_data["tokens"]))
        self.assertLess(ans_data["end_token"], len(ans_data["tokens"]))

        unans_res = self.client.post(
            "/api/qa/distilbert",
            json={
                "question": "What is the boiling point of water on Mars?",
                "context": "The Eiffel Tower was built in Paris between 1887 and 1889 and was designed by Gustave Eiffel.",
            },
        )
        self.assertEqual(unans_res.status_code, 200)
        self.assertFalse(unans_res.json()["answerable"])
        self.assertEqual(unans_res.json()["answer"], "")

    def test_bilstm_qa_response_shape(self):
        res = self.client.post(
            "/api/qa/bilstm",
            json={
                "question": "Who invented the telephone?",
                "context": "Alexander Graham Bell invented the telephone in 1876.",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("answerable", data)
        self.assertIn("answerable_score", data)
        self.assertIn("attention_weights", data)
        self.assertGreater(len(data["ctx_tokens"]), 0)
        self.assertGreater(len(data["q_tokens"]), 0)

    def test_full_pipeline(self):
        res = self.client.post(
            "/api/pipeline",
            json={"query": "What is the theory of relativity?", "top_k": 3},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["retrieved"]), 3)
        self.assertIn("best_qa", data)
        self.assertGreater(data["total_time_ms"], 0)
        for item in data["retrieved"]:
            self.assertIn("summary", item)
            self.assertIn("qa_distilbert", item)
            self.assertIn("qa_bilstm", item)


if __name__ == "__main__":
    unittest.main()
