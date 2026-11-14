"""CPU-only regression checks; no checkpoints, datasets or Torch import required.

The lightweight integration tests compile the actual inference methods from their
source AST to avoid importing the GPU stack and loading upstream face templates.
They exercise scoring, not the neural networks or end-to-end video accuracy.
"""
import ast
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "service" / "app"), str(ROOT / "service" / "ens")]
from reliability import (InvalidScores, finite_array, fusion_logit, logit, mean_probability,
                         probability, readout_shift, sigmoid, track_candidate,
                         validate_fusion, validate_readout)
from selection import validate_split_rows, validation_score


def source_objects(path, cls=None, names=()):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    if cls:
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
        tree.body = [ast.ClassDef(name=cls, bases=[], keywords=[], body=body, decorator_list=[])]
    else:
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(globals(), sig=lambda x: float(sigmoid(x)), CERT_MARGIN=0, CERT_AGREE=0)
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), ns)
    return ns


def fusion():
    return dict(members=["dfd", "pe"], coef=[0.5, 1.5], intercept=-0.2,
                detectors={"pe": {"type": "gend", "backbone": "pe"}})


class NumericTests(unittest.TestCase):
    def test_extreme_logits_are_stable(self):
        with np.errstate(over="raise", invalid="raise"):
            np.testing.assert_array_equal(sigmoid([-1e4, 0, 1e4]), [0, 0.5, 1])

    def test_finite_predictions_match_previous_formula(self):
        z = np.linspace(-25, 25, 40).reshape(10, 4)
        np.testing.assert_allclose(sigmoid(z), 1 / (1 + np.exp(-z)), atol=1e-15)
        self.assertAlmostEqual(mean_probability(z), float((1 / (1 + np.exp(-z))).mean()))

    def test_invalid_scores_never_become_real(self):
        for value in ([], [float("nan")], [float("inf")], [-float("inf")]):
            with self.subTest(value=value), self.assertRaises(InvalidScores):
                mean_probability(value)
        for value in (-0.1, 1.1, float("nan"), float("inf"), [0.5]):
            with self.subTest(value=value), self.assertRaises(InvalidScores):
                probability(value)

    def test_readout_validates_and_corrects(self):
        cfg = dict(mean=[0, 1], scale=[1, 2], coef=[[2, -1]], intercept=[0.5])
        arrays = validate_readout(cfg, outputs=1, features=2)
        np.testing.assert_allclose(readout_shift([[1, 3]], *arrays), [[1.5]])
        for change in ({"scale": [0, 1]}, {"scale": [-1, 1]}, {"coef": [[1]]},
                       {"mean": [float("nan"), 0]}, {"intercept": 1}):
            with self.subTest(change=change), self.assertRaises(InvalidScores):
                validate_readout({**cfg, **change}, outputs=1, features=2)

    def test_fusion_requires_all_members(self):
        cfg = fusion()
        self.assertAlmostEqual(fusion_logit(cfg, {"dfd": 2, "pe": -1}), -0.7)
        for scores in ({"dfd": 2}, {"dfd": 2, "pe": float("nan")}):
            with self.assertRaises(InvalidScores):
                fusion_logit(cfg, scores)

    def test_fusion_config_rejects_corruption(self):
        cfg = fusion()
        for change in ({"members": ["dfd", "dfd"]}, {"coef": [1]}, {"intercept": float("inf")},
                       {"detectors": {}}, {"members": []}, {"coef": [1, float("nan")]}):
            with self.subTest(change=change), self.assertRaises(InvalidScores):
                validate_fusion({**cfg, **change})

    def test_fusion_overflow_is_rejected(self):
        cfg = {**fusion(), "coef": [1e308, 1e308]}
        with self.assertRaises(InvalidScores):
            fusion_logit(cfg, {"dfd": 1e308, "pe": 1e308})


class TrackingTests(unittest.TestCase):
    def test_starts_with_largest_valid_face(self):
        self.assertEqual(track_candidate([[0, 0, 10, 10], [0, 0, 30, 30]], None), 1)

    def test_follows_nearby_face(self):
        self.assertEqual(track_candidate([[100, 0, 120, 20], [2, 1, 22, 21]], [0, 0, 20, 20]), 1)

    def test_disappearing_face_does_not_jump_to_bystander(self):
        self.assertIsNone(track_candidate([[100, 0, 120, 20]], [0, 0, 20, 20]))

    def test_invalid_and_missing_detections_are_skipped(self):
        self.assertIsNone(track_candidate([], None))
        self.assertIsNone(track_candidate([[0, 0, 0, 10], [0, 0, float("nan"), 20]], None))


class LiveIntegrationTests(unittest.TestCase):
    def analyzer(self):
        ns = source_objects("service/app/pipeline.py", "Analyzer",
                            ("readout", "prob", "face_scores", "finish"))
        an = ns["Analyzer"]()
        an.mu, an.sd = np.zeros(1280), np.ones(1280)
        an.W, an.b, an.lam = np.zeros((3, 1280)), np.zeros(3), 0.5
        an.fusion, an.extra = None, {}
        return an

    def outputs(self, z=1.0):
        return {"base": [dict(d=np.full(3, z), y_s=np.zeros(1024), y_t=np.zeros(256))]}

    def test_original_finite_scores_unchanged(self):
        self.assertAlmostEqual(self.analyzer().face_scores(self.outputs())["p"], 1 / (1 + math.exp(-1)))

    def test_missing_ensemble_output_is_error(self):
        an = self.analyzer()
        an.fusion = fusion()
        with self.assertRaises(InvalidScores):
            an.face_scores(self.outputs())

    def test_nan_head_is_error(self):
        with self.assertRaises(InvalidScores):
            self.analyzer().face_scores(self.outputs(float("nan")))

    def test_incomplete_face_results_are_error(self):
        st = dict(marks={}, t0=0, clip_idx=[], crops=[object()])
        with self.assertRaises(InvalidScores):
            self.analyzer().finish(st, [])

    def test_gend_mean_frame_probability(self):
        ns = source_objects("service/app/gend_live.py", "GenDMember", ("video_logit",))
        member = ns["GenDMember"]()
        member.sc = None
        z = np.array([-3., 1., 6.])
        self.assertAlmostEqual(member.video_logit(z, np.zeros((3, 1024))),
                               logit(float((1 / (1 + np.exp(-z))).mean())))
        with self.assertRaises(InvalidScores):
            member.video_logit(z, np.zeros((2, 1024)))

    def test_explicit_missing_fusion_file_is_error(self):
        fn = source_objects("service/app/gend_live.py", names=("load_fusion",))["load_fusion"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.json"
            with patch.dict(os.environ, {"DFD_FUSION_JSON": ""}):
                self.assertIsNone(fn(path))
            with patch.dict(os.environ, {"DFD_FUSION_JSON": str(path)}):
                with self.assertRaises(FileNotFoundError):
                    fn(path)
            path.write_text(json.dumps(fusion()), encoding="utf-8")
            self.assertEqual(fn(path)["members"], ["dfd", "pe"])

    def test_shared_network_preserves_each_sampling_count(self):
        ns = source_objects("service/app/gend_live.py", names=("make_member",))
        class DummyMember:
            def __init__(self, cfg, device):
                self.cfg, self.frames, self.sc = cfg, cfg["frames"], None
                self.model = object()
        ns["GenDMember"] = DummyMember
        shared = {}
        a = ns["make_member"]({"backbone": "pe", "frames": 8}, "cpu", shared)
        b = ns["make_member"]({"backbone": "pe", "frames": 24}, "cpu", shared)
        self.assertIs(a.model, b.model)
        self.assertEqual((a.frames, b.frames), (8, 24))


class TrainingTests(unittest.TestCase):
    def test_checkpoint_selection_matches_app_threshold(self):
        a = {c: dict(auc=0.99, bacc=0.60) for c in ("clean", "degraded")}
        b = {c: dict(auc=0.95, bacc=0.85) for c in ("clean", "degraded")}
        self.assertGreater(validation_score(b, "bacc"), validation_score(a, "bacc"))
        self.assertGreater(validation_score(a), validation_score(b))

    def test_bad_validation_metrics_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            r = {c: dict(auc=value) for c in ("clean", "degraded")}
            with self.assertRaises(ValueError):
                validation_score(r)

    def test_split_leakage_rejected(self):
        train = [dict(vid="r1", label=0), dict(vid="f1", label=1)]
        val = [dict(vid="r2", label=0), dict(vid="f2", label=1)]
        validate_split_rows(train, val)
        with self.assertRaises(ValueError):
            validate_split_rows(train, train)
        with self.assertRaises(ValueError):
            validate_split_rows(train, [dict(vid="r2", label=0)])
        train[0]["group"], val[1]["group"] = "same-person", "same-person"
        with self.assertRaises(ValueError):
            validate_split_rows(train, val)

    def test_zero_worker_mode_is_supported(self):
        fn = source_objects("service/ens/train_gend.py", names=("loader_options",))["loader_options"]
        with patch.dict(os.environ, {"WORKERS": "0"}):
            self.assertEqual(fn(), dict(num_workers=0))
        with patch.dict(os.environ, {"WORKERS": "2"}):
            self.assertEqual(fn(), dict(num_workers=2, persistent_workers=True, prefetch_factor=4))


if __name__ == "__main__":
    unittest.main()
