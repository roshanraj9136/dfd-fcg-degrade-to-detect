# Fine-tuning DFD-FCG on newer data: plan and decision rule

Written on 13 Nov 2026, 5:20 PM, before any test result of the fine-tuned model existed.

## What is trained
The authors' side network and heads (0.25 M parameters). CLIP ViT-L/14 stays frozen, as in the paper. The loss is
the authors' own: focal loss on the three heads plus the face-feature-guided attention loss. 800 steps of 8 clips,
AdamW, learning rate 5e-5 with warm-up and cosine decay, augmentation by mirroring, JPEG, blur, down-scaling, noise
and brightness. Code: `service/train/train.py`.

## Data (`service/train/make_sets.py`, `make_sets_gated.py`)
Training: FF++ train split (120 real, 120 fake), Celeb-DF v2 (non-test videos), MAVOS-DD train split (2025,
modern face swaps and talking-head animation), SynthForensics train split (2026, AI-generated people), Counterfactual
DeepFake Bench groups 1-24 (2026, AI-generated talking heads), social-media deepfakes (2026, training part).
Checkpoint selection: the validation splits of the same datasets only.

Never used for training or selection: the 225 reported test videos (FF++ test, FaceShifter, DFDC part 0), our fit
and selection videos, all of DFDC (its parts can share actors with the DFDC test part), and every test split below.

## How it is tested
1. The 225 reported test videos, research protocol (cached research face geometry, 2 clips, the authors' damage code),
   website options: as it is, strong blur (GB 5), heavy compression (VC 5). Both models with their own
   self-calibration read-out, refitted with exactly the research recipe on the same 49 fit videos.
2. Held-out test splits of the newer datasets (400 videos): Celeb-DF v2 official test list, social-media test part,
   MAVOS-DD test (in-domain methods) and test open-set (hififace, sonic, roop: never in training),
   SynthForensics test split, Counterfactual groups 31-36. Live-checker layout (8 clips, up to 2 faces).

## Decision rule for the live checker (fixed now)
The fine-tuned model replaces the released one on the website only if all three hold:
- on the newer test sets, mean balanced accuracy (sets with real and fake videos) improves by at least 5 points;
- on the FF++ reported test videos, balanced accuracy over the three website options drops by at most 2 points;
- on DFDC (reported test, part 0), balanced accuracy over the three options drops by at most 2 points.
Otherwise the website keeps the released model, and the result is reported either way.
