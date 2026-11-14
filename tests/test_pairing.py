"""Pair preparation must not import held-out videos into training."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service" / "ens"))
from prepare_pairs import counterpart, eligible_pairs
from sampling import PairedSampler


class PairPreparationTests(unittest.TestCase):
    def fake(self, vid="FF_DF/001_002"):
        return dict(vid=vid, set="train", source="ffpp", label=1, repo="example/ffpp")

    def test_pairs_use_the_target_video_not_identity_donor(self):
        self.assertEqual(counterpart("FF_DF/001_002"), "FF_REAL/001")
        with self.assertRaises(ValueError):
            counterpart("DFDC_F/abc")

    def test_reserved_or_validation_reals_are_never_added(self):
        fake = self.fake()
        for meta, official, reserved in (({}, {"001"}, {"001"}),
                                         ({"FF_REAL/001": dict(set="val", label=0)}, {"001"}, set()),
                                         ({}, {"002"}, set())):
            pairs, reals, rejected = eligible_pairs([fake], meta, official, reserved)
            self.assertEqual((pairs, reals, rejected), ([], [], [fake["vid"]]))

    def test_multiple_manipulations_share_one_real_download(self):
        fakes = [self.fake(), self.fake("FF_FS/001_002")]
        pairs, reals, rejected = eligible_pairs(fakes, {}, {"001", "002"}, set())
        self.assertEqual((len(pairs), len(reals), len(rejected)), (2, 1, 0))
        self.assertEqual(reals[0]["set"], "train")
        self.assertTrue(reals[0]["path"].endswith("original/001.mp4"))

    def test_fake_must_already_belong_to_training(self):
        with self.assertRaises(ValueError):
            eligible_pairs([{**self.fake(), "set": "test"}], {}, {"001"}, set())


class PairedSamplingTests(unittest.TestCase):
    def rows(self):
        return [dict(vid="real", set="train", source="ffpp", label=0),
                dict(vid="fake", set="train", source="ffpp", label=1)]

    def test_counterparts_share_augmentation_and_relative_time(self):
        sampler = PairedSampler(self.rows(), [dict(real_vid="real", fake_vid="fake")], 1)
        for i in range(0, 100, 2):
            real, fake = sampler.plan(i), sampler.plan(i + 1)
            self.assertEqual((real[0]["label"], fake[0]["label"]), (0, 1))
            self.assertEqual(real[1:], fake[1:])
            self.assertTrue(real[3])
            self.assertEqual(sampler.plan(i), real)

    def test_missing_crop_is_counted_and_uses_unpaired_fallback(self):
        sampler = PairedSampler(self.rows(), [dict(real_vid="missing", fake_vid="fake")], 1)
        self.assertEqual(sampler.unavailable, ["fake"])
        self.assertFalse(sampler.plan(0)[3])
        self.assertNotEqual(sampler.plan(0)[1:3], sampler.plan(1)[1:3])

    def test_test_rows_and_reversed_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            PairedSampler([{**self.rows()[0], "set": "test"}, self.rows()[1]], [], 1)
        with self.assertRaises(ValueError):
            PairedSampler(self.rows(), [dict(real_vid="fake", fake_vid="real")], 1)


if __name__ == "__main__":
    unittest.main()
