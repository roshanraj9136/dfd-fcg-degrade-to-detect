"""Deterministic real/fake pairs, with balanced-source fallback for unpaired data."""
import random


class PairedSampler:
    def __init__(self, rows, pairs, seed):
        self.by = {0: {}, 1: {}}
        lookup = {}
        for row in rows:
            if row.get("set") != "train" or row["label"] not in (0, 1) or row["source"] == "dfdc":
                raise ValueError("paired sampler accepts training rows only, excluding DFDC")
            if row["vid"] in lookup:
                raise ValueError("duplicate training video ID")
            lookup[row["vid"]] = row
            self.by[row["label"]].setdefault(row["source"], []).append(row)
        if not all(self.by.values()):
            raise ValueError("paired sampling needs real and fake videos")
        self.real_of, self.unavailable = {}, []
        for pair in pairs:
            real, fake = lookup.get(pair["real_vid"]), lookup.get(pair["fake_vid"])
            if real is None or fake is None:
                self.unavailable.append(pair["fake_vid"])
                continue
            if real["label"] != 0 or fake["label"] != 1 or real["source"] != fake["source"]:
                raise ValueError("pair labels or sources disagree")
            if fake["vid"] in self.real_of and self.real_of[fake["vid"]]["vid"] != real["vid"]:
                raise ValueError("fake has conflicting real counterparts")
            self.real_of[fake["vid"]] = real
        self.seed = seed

    def plan(self, index):
        """Adjacent indices form a balanced pair. Known pairs share position and augmentation seed."""
        rng = random.Random(self.seed * 1_000_003 + index // 2)
        source = rng.choice(sorted(self.by[1]))
        fake = rng.choice(self.by[1][source])
        real = self.real_of.get(fake["vid"])
        paired = real is not None
        if not paired:
            real_source = rng.choice(sorted(self.by[0]))
            real = rng.choice(self.by[0][real_source])
        real_position, real_seed = rng.random(), rng.randrange(1 << 30)
        fake_position, fake_seed = (real_position, real_seed) if paired else (rng.random(), rng.randrange(1 << 30))
        return (real, real_position, real_seed, paired) if index % 2 == 0 else (fake, fake_position, fake_seed, paired)
