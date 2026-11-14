# Handoff: DFD-FCG project (updated 14 Nov 2026, 7:10 AM)

**Status:** round 2 deployed. The live checker runs DFD-FCG + GenD-PE (released and retrained), self-calibration on
each, and a combiner; it always answers REAL or FAKE with a confidence. Viva site has the "Final system" section.
Code and results pushed: commit 000773d on github.com/roshanraj9136/dfd-fcg-degrade-to-detect.

## Live parts
- Checker site: https://dfd-fcg-degrade-to-detect.vercel.app (`web/checker`; deploy: `vercel deploy --prod --yes --project dfd-fcg-degrade-to-detect`)
- Viva site: https://dfd-fcg-viva.vercel.app (`web/viva`; deploy: `vercel deploy --prod --yes --project dfd-fcg-viva`)
- Backend: HF Space roshan9136/dfd-fcg-checker (ZeroGPU). Build `service/space/build.ps1`, upload `scratch/make_space.py --upload --msg "..."`.
  Rollback: the Space commit before round 2 is in `D:\dfd-data\ens\space_previous_sha.txt` (fed65ee0...). Removing
  `service/app/fusion.json` and re-uploading also gives the old single-detector behaviour.

## Round 2 (results/ensemble/)
- Plan, amendments and both decision records: `results/ensemble/PLAN.md` (written before the respective test results).
- Numbers: `results/ensemble/results.json` (chosen option), `all_options_round2.json` / `all_options_round2b.json`
  (every member and fusion, with validation CV scores).
- Live option: `dfd+pe_ft+pe_ft_sc+pe+pe_sc`, combiner fitted on clean + blurred + compressed validation videos.
  Held-out tests: newer sets 62.2% -> 85.0% balanced accuracy (+22.8, 95% range +17.4 to +28.3), FF++ 84.6% -> 79.0%
  (-5.6), DFDC 58.8% -> 60.4%. All 705 test videos as uploaded: 63.1% -> 85.1% correct.
- **The pre-registered switch rule was NOT met** (FF++ lost more than 2 points, mostly false alarms on clean FF++ real
  videos: 19% vs 1%). The owner asked for the most accurate firm answers on modern videos, so it was deployed anyway;
  the viva page and README say this openly.
- Self-calibration carried over unchanged to GenD-PE: strong-blur false alarms on real FF++ videos 86% -> 31%.

## Still running (local GPU, D:\dfd-data\ens\runner.ps1, jobs in D:\dfd-data\ens\jobs)
- `060_train_clip`: GenD-CLIP fine-tuning with 342 extra training videos (store2), then `061_score_clipft`. Not used
  live. To try it: `python service/ens/evaluate.py dfd pe_ft pe_ft_sc pe pe_sc clip_ft clip_ft_sc` with
  `VAL_CONDS=clean,blur,comp` (needs damaged-validation scores for clip_ft first: `service/ens/score_valdmg.py`,
  add a clip branch). `070_stop_keep_awake` then ends the keep-awake script.

## Machine changes made overnight (revert if wanted)
- Power plan (AC): sleep never, hibernate never, lid close = do nothing, screen off after 10 min.
  Restore e.g.: `powercfg /change standby-timeout-ac 30` and `powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1; powercfg /setactive SCHEME_CURRENT`.
- Closed apps to free the GPU: ChatGPT/Codex desktop, Phone Link, Cross-Device Resume (reopen normally).
- Windows grew the page file on C: (about 6-9 GB free during training).

## Rules
- Never commit `data/`, `.env*`, `.vercel`, tokens, DFD-FCG weights (research-only licence; private HF repo).
- GenD weights are MIT (yermandy/GenD_*); `service/app/gend_pe_ft.pt` holds only our fine-tuned LayerNorm + head tensors.
