import importlib.util
import sys
import tempfile
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

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules.setdefault("torch", torch)

    easyeditor = types.ModuleType("easyeditor")
    easyeditor.IKEHyperParams = object
    easyeditor.MEMITHyperParams = object
    easyeditor.BaseEditor = object
    sys.modules.setdefault("easyeditor", easyeditor)

    easyeditor_models = types.ModuleType("easyeditor.models")
    sys.modules.setdefault("easyeditor.models", easyeditor_models)

    ike = types.ModuleType("easyeditor.models.ike")
    ike.encode_ike_facts = lambda *args, **kwargs: None
    sys.modules.setdefault("easyeditor.models.ike", ike)

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = object
    sys.modules.setdefault("sentence_transformers", sentence_transformers)


def load_baseline_ike():
    install_stubs()
    path = Path(__file__).resolve().parents[1] / "scripts" / "baseline_ike.py"
    spec = importlib.util.spec_from_file_location("baseline_ike_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BaselineIkeCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_ike = load_baseline_ike()

    def make_hparams(self, results_dir: str):
        return types.SimpleNamespace(
            results_dir=results_dir,
            alg_name="IKE",
            sentence_model_name="sentence-transformers/all-MiniLM-L6-v2",
            device=0,
        )

    def test_embedding_path_matches_easyedit_cache_convention(self):
        hparams = self.make_hparams("/tmp/results")
        train_ds = [{"prompt": "a", "target_new": "b"} for _ in range(3)]

        path = self.baseline_ike.embedding_path(hparams, train_ds)

        self.assertEqual(
            path,
            "/tmp/results/IKE/embedding/all-MiniLM-L6-v2_list_3.pkl",
        )

    def test_existing_cache_skips_sentence_transformer_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hparams = self.make_hparams(tmpdir)
            train_ds = [{"prompt": "a", "target_new": "b"}]
            path = Path(self.baseline_ike.embedding_path(hparams, train_ds))
            path.parent.mkdir(parents=True)
            path.write_bytes(b"cache")

            def fail_sentence_transformer(*args, **kwargs):
                raise AssertionError("SentenceTransformer should not load when cache exists")

            def fail_encode(*args, **kwargs):
                raise AssertionError("encode_ike_facts should not run when cache exists")

            self.baseline_ike.SentenceTransformer = fail_sentence_transformer
            self.baseline_ike.encode_ike_facts = fail_encode

            self.baseline_ike.ensure_ike_embeddings(hparams, train_ds, rebuild=False)

    def test_rebuild_embeddings_loads_sentence_model_and_encodes(self):
        calls = []

        class FakeSentenceModel:
            def __init__(self, name):
                calls.append(("load", name))

            def to(self, device):
                calls.append(("to", device))
                return self

        def fake_encode(sentence_model, train_ds, hparams):
            calls.append(("encode", len(train_ds), hparams.alg_name, isinstance(sentence_model, FakeSentenceModel)))

        with tempfile.TemporaryDirectory() as tmpdir:
            hparams = self.make_hparams(tmpdir)
            train_ds = [{"prompt": "a", "target_new": "b"} for _ in range(2)]

            self.baseline_ike.SentenceTransformer = FakeSentenceModel
            self.baseline_ike.encode_ike_facts = fake_encode

            self.baseline_ike.ensure_ike_embeddings(hparams, train_ds, rebuild=True)

        self.assertEqual(
            calls,
            [
                ("load", "sentence-transformers/all-MiniLM-L6-v2"),
                ("to", "cuda:0"),
                ("encode", 2, "IKE", True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
