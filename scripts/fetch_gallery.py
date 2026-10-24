"""Build the 1,000-image gallery: 500 COCO val2017 + 500 TextCaps validation images.

Standard library only, so it runs with `uv run python scripts/fetch_gallery.py ...`.

Subcommands (run in this order):
  select    download the two annotation files, take a seeded sample (seed 47) of
            500 images per source, write data/gallery.jsonl and data/queries.jsonl
  download  fetch the 1,000 images to data/local/images/<source>/<image_id>.jpg,
            fill in sha256 and the real pixel size, replace permanent failures
            with the next id in the same seeded order
  tagsheet  write the blind 100-query tagging sheet under data/tagging/
  (none)    with --dry-run: validate and print the plan, download no images

Images are never committed: data/local/ is git-ignored. Only ids, captions,
URLs, hashes and licence notes land in the tracked files.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import html
import json
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

SEED = 47
N_PER_SOURCE = 500
N_TAG_PER_SOURCE = 50
MAX_WORKERS = 8
RETRIES = 4
USER_AGENT = "mmsearch-gallery-fetch/1.0 (research; polite, <=8 connections)"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOCAL = DATA / "local"
ANN_DIR = LOCAL / "annotations"
IMG_DIR = LOCAL / "images"
TAG_DIR = DATA / "tagging"
GALLERY = DATA / "gallery.jsonl"
QUERIES = DATA / "queries.jsonl"
REPLACEMENTS = LOCAL / "replacements.jsonl"
MANIFEST = ANN_DIR / "MANIFEST.json"

COCO_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_MEMBER = "annotations/captions_val2017.json"
COCO_JSON = ANN_DIR / "captions_val2017.json"
COCO_ZIP = ANN_DIR / "annotations_trainval2017.zip"
TC_URL = "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json"
TC_JSON = ANN_DIR / "TextCaps_0.1_val.json"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"
TC_IMG_URL = "https://open-images-dataset.s3.amazonaws.com/train/{oid}.jpg"

SOURCES = ("coco", "textcaps")
ID_PREFIX = {"coco": "coco_", "textcaps": "tc_"}


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def norm(text: str) -> str:
    """Whitespace normalisation only: collapse runs, strip the ends."""
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def http_head(url: str, timeout: int = 30) -> tuple[int, int | None]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            return r.status, int(n) if n else None
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:  # network error: report as 0
        return 0, None


def fetch_to(url: str, dest: Path, retries: int = RETRIES, timeout: int = 60) -> int:
    """Download url to dest (via dest.part, resuming with Range). Returns bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": USER_AGENT}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                with part.open(mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            part.replace(dest)
            return dest.stat().st_size
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 416:  # range past the end: .part is complete
                part.replace(dest)
                return dest.stat().st_size
            if e.code in (400, 401, 403, 404, 410):  # permanent
                part.unlink(missing_ok=True)
                raise
        except Exception as e:
            last = e
        time.sleep(2**attempt)
    assert last is not None
    raise last


def jpeg_info(path: Path) -> dict | None:
    """Parse a JPEG header: width, height, components, EXIF orientation."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if data[:2] != b"\xff\xd8":
        return None
    info = {"width": None, "height": None, "components": None, "orientation": 1}
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seglen = int.from_bytes(data[i + 2 : i + 4], "big")
        seg = data[i + 4 : i + 2 + seglen]
        if marker == 0xE1 and seg[:6] == b"Exif\x00\x00":
            info["orientation"] = _exif_orientation(seg[6:])
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            info["height"] = int.from_bytes(seg[1:3], "big")
            info["width"] = int.from_bytes(seg[3:5], "big")
            info["components"] = seg[5]
            break
        if marker == 0xDA:
            break
        i += 2 + seglen
    return info if info["width"] else None


def _exif_orientation(tiff: bytes) -> int:
    try:
        order = "little" if tiff[:2] == b"II" else "big"
        ifd = int.from_bytes(tiff[4:8], order)
        count = int.from_bytes(tiff[ifd : ifd + 2], order)
        for k in range(count):
            e = ifd + 2 + 12 * k
            if int.from_bytes(tiff[e : e + 2], order) == 0x0112:
                return int.from_bytes(tiff[e + 8 : e + 10], order)
    except Exception:
        pass
    return 1


# ----------------------------------------------------------------------------
# annotations
# ----------------------------------------------------------------------------
def ensure_annotations() -> dict:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    for name, url, dest in (
        ("coco_zip", COCO_ZIP_URL, COCO_ZIP),
        ("textcaps_val", TC_URL, TC_JSON),
    ):
        status, length = http_head(url)
        if status != 200:
            sys.exit(f"annotation URL not reachable ({status}): {url}")
        if not dest.exists() or (length and dest.stat().st_size != length):
            print(f"downloading {url} ({length} bytes) ...", flush=True)
            fetch_to(url, dest, timeout=120)
        manifest[name] = {"url": url, "bytes": dest.stat().st_size, "sha256": sha256_file(dest)}
    if not COCO_JSON.exists():
        with (
            zipfile.ZipFile(COCO_ZIP) as z,
            z.open(COCO_MEMBER) as src,
            COCO_JSON.open("wb") as out,
        ):
            shutil.copyfileobj(src, out)
    manifest["coco_captions_val2017"] = {
        "url": COCO_ZIP_URL + "#" + COCO_MEMBER,
        "bytes": COCO_JSON.stat().st_size,
        "sha256": sha256_file(COCO_JSON),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class Catalog:
    """Per-source image records and their captions, plus the seeded id order."""

    def __init__(self) -> None:
        coco = json.loads(COCO_JSON.read_text(encoding="utf-8"))
        lic = {x["id"]: x["name"] for x in coco["licenses"]}
        caps: dict[int, list[tuple[int, str]]] = {}
        for a in coco["annotations"]:
            text = norm(a["caption"])
            if text:
                caps.setdefault(a["image_id"], []).append((a["id"], text))
        self.items: dict[str, dict[str, dict]] = {"coco": {}, "textcaps": {}}
        for im in coco["images"]:
            if im["id"] not in caps:
                continue
            iid = f"coco_{im['id']:012d}"
            self.items["coco"][iid] = {
                "file_name": im["file_name"],
                "url": COCO_IMG_URL.format(file_name=im["file_name"]),
                "width": im["width"],
                "height": im["height"],
                "licence_note": (
                    f"COCO val2017 image from Flickr ({im['flickr_url']}); Flickr licence: "
                    f"{lic.get(im['license'], im['license'])} (COCO licence id {im['license']}); "
                    "annotations CC BY 4.0"
                ),
                "queries": [(f"coco-{aid}", t) for aid, t in sorted(caps[im["id"]])],
            }

        tc = json.loads(TC_JSON.read_text(encoding="utf-8"))["data"]
        by_img: dict[str, list[dict]] = {}
        for r in tc:
            by_img.setdefault(r["image_id"], []).append(r)
        for oid, rows in by_img.items():
            refs = rows[0]["reference_strs"]
            if any(r["reference_strs"] != refs for r in rows):
                sys.exit(f"TextCaps rows disagree on reference_strs for {oid}")
            texts = [norm(t) for t in refs]
            qs = [(f"tc-{oid}-{i}", t) for i, t in enumerate(texts) if t]
            if not qs:
                continue
            self.items["textcaps"][f"tc_{oid}"] = {
                "file_name": f"{oid}.jpg",
                "url": TC_IMG_URL.format(oid=oid),
                "width": rows[0]["image_width"],
                "height": rows[0]["image_height"],
                "licence_note": (
                    "TextCaps 0.1 val (annotations CC BY 4.0); image from Open Images (train), "
                    "listed as CC BY 2.0 by Open Images; original Flickr URL "
                    f"{rows[0]['flickr_original_url']}"
                ),
                "queries": qs,
            }
        # seeded order per source: sort ids, then shuffle with a fixed seed
        self.order: dict[str, list[str]] = {}
        for s in SOURCES:
            ids = sorted(self.items[s])
            random.Random(f"{SEED}-{s}").shuffle(ids)
            self.order[s] = ids

    def gallery_row(self, source: str, iid: str) -> dict:
        it = self.items[source][iid]
        return {
            "image_id": iid,
            "source": source,
            "file_name": it["file_name"],
            "url": it["url"],
            "sha256": None,
            "width": it["width"],
            "height": it["height"],
            "licence_note": it["licence_note"],
        }

    def query_rows(self, source: str, iid: str) -> list[dict]:
        return [
            {"qid": q, "text": t, "image_id": iid, "source": source, "tag": None}
            for q, t in self.items[source][iid]["queries"]
        ]


def write_tables(gallery: list[dict], queries: list[dict]) -> None:
    gallery = sorted(gallery, key=lambda r: (SOURCES.index(r["source"]), r["image_id"]))
    present = {r["image_id"] for r in gallery}
    pos = {r["image_id"]: i for i, r in enumerate(gallery)}
    queries = sorted(
        (q for q in queries if q["image_id"] in present),
        key=lambda q: (pos[q["image_id"]], q["qid"]),
    )
    write_jsonl(GALLERY, gallery)
    write_jsonl(QUERIES, queries)


# ----------------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------------
def cmd_select(args: argparse.Namespace) -> None:
    if args.dry_run:
        for u in (COCO_ZIP_URL, TC_URL):
            print(f"HEAD {u} -> {http_head(u)}")
        print(
            f"plan: seed {SEED}, {N_PER_SOURCE} images per source "
            f"({', '.join(SOURCES)}), write {GALLERY.name} + {QUERIES.name}"
        )
        return
    manifest = ensure_annotations()
    for k, v in manifest.items():
        print(f"{k}: {v['bytes']} bytes sha256={v['sha256']}")
    cat = Catalog()
    gallery, queries = [], []
    for s in SOURCES:
        print(f"{s}: {len(cat.items[s])} eligible images (>=1 caption)")
        for iid in cat.order[s][:N_PER_SOURCE]:
            gallery.append(cat.gallery_row(s, iid))
            queries.extend(cat.query_rows(s, iid))
    write_tables(gallery, queries)
    for s in SOURCES:
        g = [r for r in gallery if r["source"] == s]
        q = [r for r in queries if r["source"] == s]
        print(f"{s}: {len(g)} images, {len(q)} queries, {len(q) / len(g):.2f} captions/image")


def local_path(row: dict) -> Path:
    return IMG_DIR / row["source"] / f"{row['image_id']}.jpg"


def fetch_one(row: dict) -> tuple[dict, str | None]:
    """Fetch one image; returns (row-with-hash-and-size, error-or-None)."""
    dest = local_path(row)
    try:
        if not dest.exists() or jpeg_info(dest) is None:
            fetch_to(row["url"], dest)
        info = jpeg_info(dest)
        if info is None:
            dest.unlink(missing_ok=True)
            return row, "not a decodable JPEG header"
        out = dict(row)
        out["sha256"] = sha256_file(dest)
        out["width"], out["height"] = info["width"], info["height"]
        out["_components"], out["_orientation"] = info["components"], info["orientation"]
        return out, None
    except Exception as e:
        return row, f"{type(e).__name__}: {e}"


def cmd_download(args: argparse.Namespace) -> None:
    gallery = read_jsonl(GALLERY)
    queries = read_jsonl(QUERIES)
    validate(gallery, queries)
    if args.dry_run:
        have = sum(local_path(r).exists() for r in gallery)
        print(
            f"plan: {len(gallery)} images, {have} already on disk under {IMG_DIR}, "
            f"workers={MAX_WORKERS}, retries={RETRIES}"
        )
        rng = random.Random(SEED)
        for s in SOURCES:
            rows = [r for r in gallery if r["source"] == s]
            for r in rng.sample(rows, 3):
                print(f"HEAD {r['url']} -> {http_head(r['url'])}")
        return

    cat = Catalog()
    t0 = time.time()
    replacements: list[dict] = read_jsonl(REPLACEMENTS) if REPLACEMENTS.exists() else []
    burned = {x["failed"] for x in replacements}
    done: dict[str, dict] = {}
    seen_sha: dict[str, str] = {}
    pending = list(gallery)
    total_bytes = 0
    while pending:
        failures: list[tuple[dict, str]] = []
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for n, (row, err) in enumerate(ex.map(fetch_one, pending), 1):
                if err is None:
                    done[row["image_id"]] = row
                else:
                    failures.append((row, err))
                if n % 100 == 0:
                    print(f"  {n}/{len(pending)} fetched", flush=True)
        # duplicate-content check across the whole gallery (first id in sorted order wins)
        seen_sha.clear()
        for iid in sorted(done):
            h = done[iid]["sha256"]
            if h in seen_sha:
                failures.append((done.pop(iid), f"duplicate of {seen_sha[h]} (same sha256)"))
            else:
                seen_sha[h] = iid
        pending = []
        for row, err in failures:
            src = row["source"]
            burned.add(row["image_id"])
            used = set(done) | burned | {r["image_id"] for r in pending}
            nxt = next(i for i in cat.order[src] if i not in used)
            rep = cat.gallery_row(src, nxt)
            replacements.append(
                {"source": src, "failed": row["image_id"], "reason": err, "replacement": nxt}
            )
            print(f"  replaced {row['image_id']} ({err}) -> {nxt}")
            pending.append(rep)
            local_path(row).unlink(missing_ok=True) if "duplicate" in err else None
        if replacements:
            write_jsonl(REPLACEMENTS, replacements)

    final_gallery = []
    final_queries: list[dict] = []
    stats = {"grayscale": 0, "rotated_exif": 0}
    for iid, row in done.items():
        total_bytes += local_path(row).stat().st_size
        stats["grayscale"] += row.get("_components") == 1
        stats["rotated_exif"] += row.get("_orientation", 1) != 1
        final_gallery.append({k: v for k, v in row.items() if not k.startswith("_")})
        final_queries.extend(cat.query_rows(row["source"], iid))
    for s in SOURCES:
        assert sum(r["source"] == s for r in final_gallery) == N_PER_SOURCE, s
    write_tables(final_gallery, final_queries)
    sizes = sorted(max(r["width"], r["height"]) for r in final_gallery)
    print(
        f"done: {len(final_gallery)} images, {total_bytes} bytes, "
        f"{len(replacements)} replacements, "
        f"{time.time() - t0:.0f}s"
    )
    print(
        f"grayscale={stats['grayscale']} exif-rotated={stats['rotated_exif']} "
        f"longest side min/median/max={sizes[0]}/{sizes[len(sizes) // 2]}/{sizes[-1]}"
    )


def validate(gallery: list[dict], queries: list[dict]) -> None:
    ids = [r["image_id"] for r in gallery]
    assert len(ids) == len(set(ids)), "duplicate image_id"
    for s in SOURCES:
        rows = [r for r in gallery if r["source"] == s]
        assert len(rows) == N_PER_SOURCE, f"{s}: {len(rows)} images"
        assert all(r["image_id"].startswith(ID_PREFIX[s]) for r in rows), f"{s}: id prefix"
    qids = [q["qid"] for q in queries]
    assert len(qids) == len(set(qids)), "duplicate qid"
    idset = set(ids)
    assert all(q["image_id"] in idset for q in queries), "query points at unknown image"
    assert idset == {q["image_id"] for q in queries}, "image without queries"
    hashes = [r["sha256"] for r in gallery if r["sha256"]]
    assert len(hashes) == len(set(hashes)), "duplicate sha256"
    print(
        f"ok: {len(gallery)} images, {len(queries)} queries; "
        f"{len(hashes)} hashed; ids unique, prefixes disjoint"
    )


def cmd_tagsheet(args: argparse.Namespace) -> None:
    gallery = read_jsonl(GALLERY)
    queries = read_jsonl(QUERIES)
    validate(gallery, queries)
    rng = random.Random(f"{SEED}-tagsheet")
    picked: list[dict] = []
    for s in SOURCES:
        imgs = sorted(r["image_id"] for r in gallery if r["source"] == s)
        for iid in rng.sample(imgs, N_TAG_PER_SOURCE):  # 50 distinct images, 1 query each
            qs = sorted((q for q in queries if q["image_id"] == iid), key=lambda q: q["qid"])
            picked.append(rng.choice(qs))
    rng.shuffle(picked)
    by_id = {r["image_id"]: r for r in gallery}
    img_out = LOCAL / "tagging_images"
    img_out.mkdir(parents=True, exist_ok=True)
    TAG_DIR.mkdir(parents=True, exist_ok=True)
    rows, key = [], []
    for n, q in enumerate(picked, 1):
        blind = f"t{n:03d}"
        shutil.copyfile(local_path(by_id[q["image_id"]]), img_out / f"{blind}.jpg")
        rows.append(
            {
                "qid": blind,
                "text": q["text"],
                "image_path": f"../local/tagging_images/{blind}.jpg",
                "tag": "",
            }
        )
        key.append({"blind_id": blind, "qid": q["qid"]})
    with (TAG_DIR / "tag_sheet.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "text", "image_path", "tag"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with (TAG_DIR / "tag_key.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "qid"], lineterminator="\n")
        w.writeheader()
        w.writerows(key)
    cards = "\n".join(
        f'<section><h2>{r["qid"]}</h2><img src="{r["image_path"]}" alt="{r["qid"]}" loading="lazy">'
        f"<p>{html.escape(r['text'])}</p></section>"
        for r in rows
    )
    (TAG_DIR / "tag_sheet.html").write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        "<title>Tagging sheet</title><style>"
        "body{font:16px/1.4 system-ui,sans-serif;max-width:900px;margin:1rem auto;padding:0 1rem}"
        "section{display:flex;gap:1rem;align-items:flex-start;"
        "border-top:1px solid #ccc;padding:1rem 0}"
        "h2{font-size:1rem;min-width:3.5rem;margin:0}img{max-width:420px;max-height:320px;height:auto}"
        "p{margin:0;font-size:1.1rem}</style></head><body>\n"
        "<h1>Tag each caption: visual, text or both</h1>\n"
        "<p>Record tags in tag_sheet.csv, matching the id shown here "
        "(see README.md for the labels).</p>\n"
        f"{cards}\n</body></html>\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} sheet rows ({N_TAG_PER_SOURCE} per source) to {TAG_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="validate and print the plan; download no images"
    )
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("select")
    sub.add_parser("download")
    sub.add_parser("tagsheet")
    args = ap.parse_args()
    if args.cmd == "select":
        cmd_select(args)
    elif args.cmd == "download":
        cmd_download(args)
    elif args.cmd == "tagsheet":
        cmd_tagsheet(args)
    elif args.dry_run:
        # no subcommand: validate whatever exists and show the whole plan
        cmd_select(args)
        if GALLERY.exists():
            cmd_download(args)
        else:
            print("no data/gallery.jsonl yet: run `select` first")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
