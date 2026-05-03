import importlib.util
import sys
import types
import unittest
from pathlib import Path


def install_stubs() -> None:
    def equal(left, right):
        if isinstance(left, list) and isinstance(right, list):
            return [equal(l_item, r_item) for l_item, r_item in zip(left, right)]
        return left == right

    def flatten(values):
        if isinstance(values, list):
            out = []
            for value in values:
                out.extend(flatten(value))
            return out
        return [values]

    numpy = types.ModuleType("numpy")
    numpy.equal = equal
    numpy.mean = lambda values: sum(flatten(values)) / len(flatten(values))
    sys.modules.setdefault("numpy", numpy)

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: NoGrad()
    sys.modules.setdefault("torch", torch)

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    sys.modules.setdefault("transformers", transformers)

    easyeditor = types.ModuleType("easyeditor")
    easyeditor.IKEHyperParams = object
    easyeditor.MEMITHyperParams = object
    easyeditor.BaseEditor = object
    sys.modules.setdefault("easyeditor", easyeditor)

    easyeditor_models = types.ModuleType("easyeditor.models")
    sys.modules.setdefault("easyeditor.models", easyeditor_models)

    memit = types.ModuleType("easyeditor.models.memit")
    memit.apply_memit_to_model = lambda *args, **kwargs: (None, {})
    sys.modules.setdefault("easyeditor.models.memit", memit)

    evaluate_pkg = types.ModuleType("easyeditor.evaluate")
    sys.modules.setdefault("easyeditor.evaluate", evaluate_pkg)

    evaluate = types.ModuleType("easyeditor.evaluate.evaluate")
    evaluate.compute_edit_quality = lambda *args, **kwargs: {}
    sys.modules.setdefault("easyeditor.evaluate.evaluate", evaluate)

    util = types.ModuleType("easyeditor.util")
    util.nethook = types.SimpleNamespace(get_parameter=lambda model, name: None)
    sys.modules.setdefault("easyeditor.util", util)


def load_batch_memit():
    install_stubs()
    path = Path(__file__).resolve().parents[1] / "scripts" / "batch_memit.py"
    spec = importlib.util.spec_from_file_location("batch_memit_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BatchMemitMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_memit = load_batch_memit()

    def test_records_to_requests_preserves_easyedit_eval_fields(self):
        records = [
            {
                "prompt": "Subject relation",
                "subject": "Subject",
                "target_new": "New",
                "ground_truth": "Old",
                "rephrase_prompt": "Rephrased relation",
                "locality_prompt": "Unrelated relation",
                "locality_ground_truth": "Unrelated",
            }
        ]

        request = self.batch_memit.records_to_requests(records)[0]

        self.assertEqual(request["prompt"], "Subject relation")
        self.assertEqual(request["subject"], "Subject")
        self.assertEqual(request["target_new"], "New")
        self.assertEqual(request["ground_truth"], "Old")
        self.assertEqual(request["rephrase_prompt"], "Rephrased relation")
        self.assertEqual(request["portability"], {})
        self.assertEqual(
            request["locality"],
            {
                "neighborhood": {
                    "prompt": "Unrelated relation",
                    "ground_truth": "Unrelated",
                }
            },
        )

    def test_summarize_uses_post_rewrite_and_pre_post_locality_preservation(self):
        pre_metrics = [
            {"locality": {"neighborhood_output": [[1, 2], [3]]}},
            {"locality": {"neighborhood_output": [[9], [8, 7]]}},
        ]
        post_metrics = [
            {
                "rewrite_acc": [1.0],
                "rephrase_acc": [0.5],
                "locality": {"neighborhood_output": [[1, 2], [4]]},
            },
            {
                "rewrite_acc": [0.0],
                "rephrase_acc": [1.0],
                "locality": {"neighborhood_output": [[9], [8, 0]]},
            },
        ]

        summary = self.batch_memit.summarize(pre_metrics, post_metrics)

        self.assertEqual(summary["rewrite_acc"], 0.5)
        self.assertEqual(summary["rephrase_acc"], 0.75)
        self.assertEqual(summary["locality_acc"], 0.625)

    def test_flatten_handles_lists_scalars_and_empty_values(self):
        self.assertEqual(self.batch_memit.flatten([1.0, 0.0]), 0.5)
        self.assertEqual(self.batch_memit.flatten(1), 1.0)
        self.assertIsNone(self.batch_memit.flatten([]))
        self.assertIsNone(self.batch_memit.flatten({"not": "metric"}))


if __name__ == "__main__":
    unittest.main()
