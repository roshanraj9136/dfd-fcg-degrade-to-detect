# Paper draft

`main.pdf` is a 7-page preprint draft of this project: *Fake, or Just Low Quality? Decision Collapse of Deepfake
Detectors under Video Degradation, and a Training-Free Self-Calibration*.

Every number in it is generated from the result files, never typed by hand:

```
python service/ens/selfcal_transfer.py   # GenD raw vs self-calibrated, from stored scores (CPU)
python paper/make_numbers.py             # -> paper/numbers.tex, paper/tables.tex
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Before submitting

1. **Your details:** replace `[Institution]` and `[email]` in `main.tex`. Add your supervisor as co-author if they
   contributed, with their agreement.
2. **References:** entries marked `TODO` in `refs.bib` (Perception Encoder, MAVOS-DD, SynthForensics, Counterfactual
   DeepFake Bench, social-media deepfakes) need their paper citations checked on the publisher or arXiv page.
3. **Read it critically.** Check every claim against your own understanding; you will have to defend it.
4. **Where to post:** arXiv, category cs.CV (Computer Vision). First-time submitters usually need an endorsement from an
   existing arXiv author in that category; your supervisor can often provide it. arXiv is free and gives the paper a
   permanent, citable link.
5. **Workshops** that accept short papers on media forensics (for example CVPR and WACV workshops on deepfakes and
   media forensics) are a good next step after arXiv.

## What the paper claims

1. AUROC hides a decision collapse: DFD-FCG keeps AUROC 0.946 under the authors' degradations but calls 22.5% of real
   test videos fake (63% under strong blur).
2. The cause: fine natural detail counts as evidence of "real" (2-px blur probe: +1.44 on real, -0.17 on fakes).
3. Self-calibration, training-free: false alarms 22.5% -> 8.7% on unseen degradation types.
4. **New:** the collapse and the fix transfer to GenD (WACV 2026): strong-blur false alarms 86% -> 31% (PE), 96% -> 30% (CLIP).
5. The fused public checker: 85.1% correct on 705 held-out videos (63.1% before), with the FF++ trade-off reported.
