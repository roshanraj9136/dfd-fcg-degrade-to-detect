# Accuracy study for the live checker (version 2)

Chooses the website's settings on videos that are in no reported result, then tests them once on the reported test
videos. Results: `results/accuracy_study.md` (generated) and the raw per-clip scores in `results/accuracy_study/`.

```bash
# 1. selection set: 80 unused FF++ test videos + 20 new DFDC pairs  -> data/manifest_select.csv
python service/study/select_set.py
# 2. per-clip scores for every setting (8 clips / 10 s and 2 clips / 144 frames, up to 2 faces, normal + mirrored)
STUDY_PROBE=0 python service/study/study.py ../data/manifest_select.csv select results/accuracy_study/study_select.jsonl
# 3. choose the setting and the "sure" margin on the selection set only
python service/study/study_report.py results/accuracy_study/study_select.jsonl --cert new,0,2
# 4. then, once, the reported test videos (with the blur test, for strict mode)
STUDY_PROBE=1 python service/study/study.py ../data/manifest.csv test,cross results/accuracy_study/study_test.jsonl
# 5. numbers for the viva site and the markdown report
python service/study/upgrade_summary.py results/accuracy_study/study_test.jsonl results/accuracy_study/study_select.jsonl \
       new 0 2 0.16 0 results/accuracy_study/upgrade.json
python service/study/write_md.py
```

The scoring in `study_report.py` (`face_p`, `video_p`) uses the same formulas as `Analyzer.finish` in
`service/app/pipeline.py`: per clip the mean of the three head sigmoids of `d - g`, averaged over clips (and views),
the most suspicious face decides. Face finding runs on the GPU in the study (fp32, video scores within 0.002 of the
Space's CPU face finding, checked on 2 videos); on a 4 GB laptop GPU the test run takes about 2 hours.
