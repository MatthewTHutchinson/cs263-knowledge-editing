import unittest

from scripts import prepare_counterfact_original as prep


class PrepareCounterFactOriginalTests(unittest.TestCase):
    def test_convert_record_uses_original_paraphrase_and_neighborhood(self):
        record = {
            "case_id": 7,
            "pararel_idx": 11,
            "requested_rewrite": {
                "prompt": "The headquarters of {} is in",
                "relation_id": "P159",
                "subject": "Acme",
                "target_new": {"str": "Berlin", "id": "Q64"},
                "target_true": {"str": "Paris", "id": "Q90"},
            },
            "paraphrase_prompts": [
                "Unrelated retrieved sentence. Acme's global headquarters is located in",
                "Acme is headquartered in",
            ],
            "neighborhood_prompts": ["The headquarters of Neighbor Corp is in"],
        }

        converted = prep.convert_record(record)

        self.assertIsNotNone(converted)
        self.assertEqual(converted["prompt"], "The headquarters of Acme is in")
        self.assertEqual(converted["subject"], "Acme")
        self.assertEqual(converted["target_new"], "Berlin")
        self.assertEqual(converted["ground_truth"], "Paris")
        self.assertEqual(converted["rephrase_prompt"], "Acme is headquartered in")
        self.assertEqual(converted["locality_prompt"], "The headquarters of Neighbor Corp is in")
        self.assertEqual(converted["locality_ground_truth"], "Paris")
        self.assertEqual(converted["relation_id"], "P159")

    def test_convert_record_skips_missing_eval_prompts(self):
        record = {
            "requested_rewrite": {
                "prompt": "{} plays",
                "subject": "Player",
                "target_new": {"str": "basketball"},
                "target_true": {"str": "football"},
            },
            "paraphrase_prompts": [],
            "neighborhood_prompts": ["Neighbor plays"],
        }

        self.assertIsNone(prep.convert_record(record))


if __name__ == "__main__":
    unittest.main()
