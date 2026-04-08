from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hackathon_ai_env.web import DashboardState


class DashboardStateTest(unittest.TestCase):
    def test_train_builds_policy_summary(self) -> None:
        state = DashboardState(default_episodes=6, seed=5)
        payload = state.train(episodes=6)
        self.assertTrue(payload["status"]["trained"])
        self.assertEqual(payload["status"]["trained_episodes"], 6)
        self.assertGreater(len(payload["timeline"]), 0)
        self.assertGreater(len(payload["status"]["training_timeline"]), 0)
        self.assertIn("policy", payload["status"])
        self.assertIn("q_table", payload["status"]["policy"])

    def test_ask_returns_answer_payload(self) -> None:
        state = DashboardState(default_episodes=6, seed=5)
        payload = state.ask(
            query="What pricing model should we use for a student startup?",
            episodes=6,
            warm_memory=True,
        )
        self.assertIn("result", payload)
        self.assertTrue(payload["result"]["answer"])
        self.assertIn("memory_bank", payload["status"]["policy"])
        self.assertTrue(payload["status"]["feedback_pending"])
        self.assertEqual(
            payload["status"]["last_inference"]["query"],
            "What pricing model should we use for a student startup?",
        )

    def test_feedback_submission_updates_status(self) -> None:
        state = DashboardState(default_episodes=6, seed=5)
        state.ask(
            query="Suggest a startup idea under 1 lakh.",
            episodes=6,
            warm_memory=True,
        )
        payload = state.submit_feedback(rating=4, notes="Good direction, add more specific niches.")
        self.assertFalse(payload["status"]["feedback_pending"])
        self.assertEqual(payload["feedback"]["rating"], 4)
        self.assertIn("reward", payload["feedback"])

    def test_retraining_preserves_user_feedback_memory(self) -> None:
        state = DashboardState(default_episodes=6, seed=5)
        query = "What is the recipe of paneer?"
        feedback = (
            "Saute onions, ginger-garlic paste, tomatoes, and spices in oil and ghee until aromatic. "
            "Add cubed paneer and simmer for 5 minutes, then finish with kasuri methi and coriander."
        )
        state.ask(query=query, episodes=6, warm_memory=True)
        state.submit_feedback(rating=5, notes=feedback)
        state.train(episodes=6)
        payload = state.ask(query=query, episodes=6, warm_memory=True)
        self.assertIn("paneer", payload["result"]["answer"].lower())
        self.assertIn("saute onions", payload["result"]["answer"].lower())
        self.assertIn("simmer for 5 minutes", payload["result"]["answer"].lower())
        self.assertNotIn("1 to 2 cups paneer", payload["result"]["answer"].lower())

    def test_reset_clears_latest_inference(self) -> None:
        state = DashboardState(default_episodes=6, seed=5)
        state.ask(
            query="Today is raining, what should I drink?",
            episodes=6,
            warm_memory=True,
        )
        payload = state.reset()
        self.assertIsNone(payload["status"]["last_inference"])
        self.assertFalse(payload["status"]["feedback_pending"])
        self.assertEqual(payload["status"]["training_timeline"], [])

    def test_dashboard_reload_restores_saved_policy_and_memory(self) -> None:
        query = "What is the recipe of paneer?"
        feedback = (
            "Saute onions, ginger-garlic paste, tomatoes, and spices in oil and ghee until aromatic. "
            "Add cubed paneer and simmer for 5 minutes, then finish with kasuri methi and coriander."
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "dashboard-state.json"
            state = DashboardState(default_episodes=6, seed=5, storage_path=state_path)
            state.ask(query=query, episodes=6, warm_memory=True)
            state.submit_feedback(rating=5, notes=feedback)
            state.train(episodes=6)
            self.assertTrue(state_path.exists())

            restored = DashboardState(default_episodes=6, seed=5, storage_path=state_path)
            status = restored.status()["status"]
            self.assertTrue(status["trained"])
            self.assertEqual(status["trained_episodes"], 6)
            self.assertGreater(status["q_state_count"], 0)

            payload = restored.ask(query=query, episodes=6, warm_memory=True)
            self.assertIn("paneer", payload["result"]["answer"].lower())
            self.assertIn("saute onions", payload["result"]["answer"].lower())


if __name__ == "__main__":
    unittest.main()
