"""Selection protocol checks that do not need any dataset or model weights."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service" / "ens"))
import select_clip_fusion as selection


class ClipSelectionTests(unittest.TestCase):
    def test_equal_complete_cohort_and_reserved_sources(self):
        meta = {
            "complete": dict(set="val", source="ffpp", label=0),
            "partial": dict(set="val", source="mavos", label=1),
            "reserved": dict(set="val", source="synth", label=1),
            "dfdc": dict(set="val", source="dfdc", label=0),
            "test": dict(set="test", source="mavos", label=1),
        }
        scores = {m: {(v, c, 0): 0.1 for v in meta for c in ("clean", "blur", "comp")}
                  for m in selection.MEMBERS}
        del scores["clip_ft"][("partial", "blur", 0)]
        with patch.object(selection, "metadata", return_value=meta), \
                patch.object(selection.Ev, "load_members", return_value=(scores, {})):
            actual, _, coverage = selection.load_complete()
        self.assertEqual(coverage["video_ids"], ["complete"])
        self.assertEqual(coverage["incomplete_videos"], ["partial"])
        for member in selection.MEMBERS:
            self.assertEqual({k[0] for k in actual[member]}, {"complete", "test"})

    def test_candidate_needs_gain_without_worse_false_alarms(self):
        base = dict(cv=dict(bacc=0.8, caught=0.8, ffpp_clean_false_alarm=0.1))
        good = dict(cv=dict(bacc=0.81, caught=0.8, ffpp_clean_false_alarm=0.1))
        self.assertTrue(selection.qualifies(good, base))
        for key, value in (("bacc", 0.801), ("caught", 0.7), ("ffpp_clean_false_alarm", 0.2)):
            bad = dict(cv={**good["cv"], key: value})
            self.assertFalse(selection.qualifies(bad, base))

    def test_frozen_weights_are_used_without_refitting(self):
        model = selection.FrozenFusion(dict(coef=[2, -1], intercept=0.5))
        np.testing.assert_allclose(model.decision_function([[1, 3], [2, 1]]), [-0.5, 3.5])

    def test_release_rejects_old_dataset_regression(self):
        base = dict(mean_new_bacc=0.85, ffpp=dict(bacc=0.79), dfdc=dict(bacc=0.60),
                    fsh=dict(bacc=0.64), synth=dict(caught=1.0), sf=dict(caught=0.76))
        candidate = {**base, "mean_new_bacc": 0.87}
        self.assertTrue(selection.release_rule(candidate, base)["metrics_pass"])
        candidate = {**candidate, "ffpp": dict(bacc=0.78)}
        self.assertFalse(selection.release_rule(candidate, base)["metrics_pass"])


if __name__ == "__main__":
    unittest.main()
