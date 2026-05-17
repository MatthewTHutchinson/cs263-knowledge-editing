from collections import Counter
import unittest

from src.probes.probe_set import EDIT_CASES, PROBES


class ProbeSetTests(unittest.TestCase):
    def test_probe_set_is_balanced_across_topics_and_categories(self):
        by_edit = Counter(p.edit_key for p in PROBES)
        by_category = Counter(p.category for p in PROBES)

        self.assertEqual(len(EDIT_CASES), 15)
        self.assertEqual(len(PROBES), 225)
        self.assertEqual(set(by_edit), set(EDIT_CASES))
        self.assertEqual(set(by_edit.values()), {15})
        self.assertEqual(
            by_category,
            {
                "logical_negation": 45,
                "symmetric_inverse": 45,
                "compositional": 45,
                "contradiction": 45,
                "chain_of_thought": 45,
            },
        )


if __name__ == "__main__":
    unittest.main()
