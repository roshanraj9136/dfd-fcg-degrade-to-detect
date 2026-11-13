"""Render the share image (og.png, 1200x630) and the phone home-screen icons from the SVGs in this folder.
python web/checker/brand/render.py      (needs playwright + Chrome)"""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent / "public"
jobs = [(HERE / "og.svg", PUBLIC / "og.png", 1200, 630),
        (PUBLIC / "favicon.svg", PUBLIC / "apple-touch-icon.png", 180, 180),
        (PUBLIC / "favicon.svg", PUBLIC / "icon-192.png", 192, 192),
        (PUBLIC / "favicon.svg", PUBLIC / "icon-512.png", 512, 512)]
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    for src, out, w, h in jobs:
        pg = b.new_page(viewport={"width": w, "height": h})
        svg = src.read_text(encoding="utf-8")
        pg.set_content(f'<html><body style="margin:0"><div style="width:{w}px;height:{h}px">'
                       f'{svg.replace("<svg ", f"<svg width=\"{w}\" height=\"{h}\" ", 1)}</div></body></html>')
        pg.screenshot(path=str(out), clip=dict(x=0, y=0, width=w, height=h))
        pg.close()
        print("wrote", out.name)
    b.close()
