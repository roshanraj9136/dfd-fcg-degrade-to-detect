"""
Load the exported results (web/frontend/public/data) into Neon Postgres.

    NEON_OWNER_URL=postgresql://... python web/api/seed.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import connect  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "frontend" / "public" / "data"


def main():
    load = lambda n: json.load(open(DATA / n))  # noqa: E731
    conn = connect(os.environ["NEON_OWNER_URL"], timeout=60)
    conn.run("begin")
    for key, fname in [("summary", "summary.json"), ("analysis", "analysis.json"),
                       ("conditions", "conditions.json"), ("projection", "projection.json"),
                       ("example", "example.json"), ("drift_layers", "drift_layers.json")]:
        conn.run("insert into kv (key, value) values (:k, cast(:v as jsonb)) "
                 "on conflict (key) do update set value = excluded.value, updated_at = now()",
                 k=key, v=json.dumps(load(fname)))
    videos = load("videos.json")
    for v in videos:
        detail = json.load(open(DATA / "maps" / (v["vid"].replace("/", "__") + ".json")))
        conn.run("insert into videos (vid, split, method, label, public_frames, scores, detail) "
                 "values (:vid, :split, :method, :label, :pub, cast(:scores as jsonb), cast(:detail as jsonb)) "
                 "on conflict (vid) do update set split=excluded.split, method=excluded.method, "
                 "label=excluded.label, public_frames=excluded.public_frames, scores=excluded.scores, "
                 "detail=excluded.detail",
                 vid=v["vid"], split=v["split"], method=v["method"], label=v["label"], pub=v["public"],
                 scores=json.dumps(v["scores"]), detail=json.dumps(detail))
    conn.run("commit")
    print("videos in database:", conn.run("select count(*) from videos")[0][0])
    conn.close()


if __name__ == "__main__":
    main()
