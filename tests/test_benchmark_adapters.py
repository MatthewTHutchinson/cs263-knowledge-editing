import unittest

from src.benchmarks import mquake, ripple_edits


class MQuAKEAdapterTests(unittest.TestCase):
    def test_record_to_eval_case_extracts_edits_and_answers(self):
        record = {
            "case_id": 7,
            "requested_rewrite": [
                {
                    "prompt": "{} is associated with the sport of",
                    "subject": "Dudley Town F.C.",
                    "target_new": {"str": "cricket"},
                    "target_true": {"str": "association football"},
                    "question": "Which sport is Dudley Town F.C. associated with?",
                }
            ],
            "questions": ["What is the capital of the country where the sport originated?"],
            "new_answer": "Oderzo",
            "new_answer_alias": ["Oderzo, Italy"],
            "new_single_hops": [{"question": "Which sport?", "answer": "cricket"}],
        }

        case = mquake.record_to_eval_case(record)

        self.assertEqual(case["case_id"], 7)
        self.assertEqual(
            case["requests"][0],
            {
                "prompt": "Dudley Town F.C. is associated with the sport of",
                "subject": "Dudley Town F.C.",
                "target_new": "cricket",
                "ground_truth": "association football",
                "question": "Which sport is Dudley Town F.C. associated with?",
            },
        )
        self.assertEqual(case["multihop_answers"], ["Oderzo", "Oderzo, Italy"])
        self.assertTrue(mquake.score_multihop_generation(record, "The answer is Oderzo."))

    def test_summarize_records_reports_distributions(self):
        summary = mquake.summarize_records(
            [
                {"requested_rewrite": [{}, {}], "questions": [1, 2, 3], "new_single_hops": [1, 2]},
                {"requested_rewrite": [{}], "questions": [1], "new_single_hops": [1]},
            ]
        )

        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["edit_count_distribution"], {1: 1, 2: 1})
        self.assertEqual(summary["question_count_distribution"], {1: 1, 3: 1})


class RippleEditsAdapterTests(unittest.TestCase):
    def test_edit_to_request_splits_declarative_fact(self):
        record = {
            "edit": {
                "prompt": "The name of the country of citizenship of Leonardo DiCaprio is Syria.",
                "original_fact": {
                    "prompt": "The name of the country of citizenship of Leonardo DiCaprio is United States of America."
                },
            }
        }

        request = ripple_edits.edit_to_request(record)

        self.assertEqual(request["prompt"], "The name of the country of citizenship of Leonardo DiCaprio is")
        self.assertEqual(request["subject"], "Leonardo DiCaprio")
        self.assertEqual(request["target_new"], "Syria")
        self.assertEqual(request["ground_truth"], "United States of America")

    def test_infer_subject_handles_subject_first_prompt(self):
        self.assertEqual(
            ripple_edits.infer_subject("Super Bowl LV is followed by"),
            "Super Bowl LV",
        )

    def test_summarize_records_counts_criteria_queries(self):
        record = {
            "example_type": "popular",
            "edit": {"relation": "COUNTRY_OF_CITIZENSHIP"},
            "Logical_Generalization": [
                {
                    "test_queries": [
                        {
                            "prompt": "Leonardo DiCaprio is a citizen of",
                            "answers": [{"value": "Syria", "aliases": ["Syrian Arab Republic"]}],
                        }
                    ]
                }
            ],
            "Relation_Specifity": [],
            "Subject_Aliasing": [],
            "Compositionality_I": [],
            "Compositionality_II": [],
            "Forgetfulness": [],
        }

        summary = ripple_edits.summarize_records([record])

        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["example_types"], {"popular": 1})
        self.assertEqual(summary["criterion_tests"]["Logical_Generalization"], 1)
        self.assertEqual(summary["criterion_queries"]["Logical_Generalization"], 1)
        query = record["Logical_Generalization"][0]["test_queries"][0]
        self.assertEqual(ripple_edits.query_answers(query), ["Syria", "Syrian Arab Republic"])
        self.assertTrue(ripple_edits.score_query_generation(query, "Syria is the answer"))


if __name__ == "__main__":
    unittest.main()
