"""Embed the gallery and its captions with CLIP, and OCR the gallery, writing arrays for replay.

Local-only stage. Everything in ``src/mmsearch`` replays the files written here, so CI never
needs a model. Run it from the project root::

    uv run --extra embed python scripts/embed_and_ocr.py --dry-run
    uv run --extra embed python scripts/embed_and_ocr.py --only all
    uv run python scripts/embed_and_ocr.py --verify          # torch-free, numpy only

Why Hugging Face ``transformers`` and not ``open_clip_torch``
--------------------------------------------------------------
Both load the same OpenAI ViT-B/32 weights (MIT licence). ``transformers`` is used because the
checkpoint (``openai/clip-vit-base-patch32``) is the OpenAI release as published by OpenAI, the
tokenizer is a fast tokenizer that can report un-truncated token lengths (needed to count how
many captions exceed CLIP's 77-token context), and it adds no vision-model dependency on
``timm``/``torchvision``: the image transform is the standard resize-bicubic / centre-crop 224 /
CLIP-mean-std recipe done with Pillow + numpy, which is what the OpenAI preprocessor does.
Torch on Windows from PyPI is the CPU build, so no custom index is needed.

Outputs (read by the package loaders)
-------------------------------------
``data/embeddings/clip_img.npy``  float32 (n_images, d), L2-normalised, rows = meta image_ids
``data/embeddings/clip_txt.npy``  float32 (n_queries, d), L2-normalised, rows = meta qids
``data/embeddings/meta.json``     model/library versions, ids, image sha256s, truncation count
``data/ocr/ocr.jsonl``            one row per image: image_id, lines[{text, confidence}], engine,
                                  preprocessing (``lines`` is empty when no text was found)
``data/ocr/ocr_meta.json``        sidecar with the sha256 each OCR row was computed from

Caching: image vectors and OCR results are cached by image sha256 under ``<data-dir>/.cache/``
(git-ignored), so a re-run on unchanged images loads instead of recomputing. OCR is the slow
part and is resumable: each finished image is appended to ``ocr.jsonl`` straight away.

OCR note: the images are natural scenes, not scanned documents, so the default preprocessing is
none (the raw image, decoded by OpenCV, which gives the BGR layout PaddleOCR expects).
``--ocr-preprocessing clahe`` applies the same CLAHE as the document scanner project, for
comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

CLIP_HF_ID = "openai/clip-vit-base-patch32"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CONTEXT_LENGTH = 77
OCR_MODELS = {
    "mobile": ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
    "server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
}
DEFAULT_OCR_MODELS = "mobile"
OCR_PREPROCESSING = ("none", "clahe")
# Fallback ETA constants (seconds per item on the dev CPU); a measured value from the previous
# run, stored in the cache, overrides them.
DEFAULT_SEC = {"clip_img": 0.10, "clip_txt": 0.01, "ocr": 1.0}


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_path(data_dir: Path, row: dict) -> Path:
    return data_dir / "local" / "images" / row["source"] / f"{row['image_id']}.jpg"


def load_inputs(data_dir: Path, limit: int | None):
    """Return (gallery rows, query rows), restricted to the first ``limit`` images."""
    gpath, qpath = data_dir / "gallery.jsonl", data_dir / "queries.jsonl"
    problems = [str(p) for p in (gpath, qpath) if not p.exists()]
    if problems:
        raise SystemExit(f"missing input file(s): {', '.join(problems)}")
    gallery, queries = read_jsonl(gpath), read_jsonl(qpath)
    ids = [r["image_id"] for r in gallery]
    if len(set(ids)) != len(ids):
        raise SystemExit("gallery.jsonl has duplicate image_id values")
    qids = [r["qid"] for r in queries]
    if len(set(qids)) != len(qids):
        raise SystemExit("queries.jsonl has duplicate qid values")
    if limit is not None:
        gallery = gallery[:limit]
        keep = {r["image_id"] for r in gallery}
        queries = [q for q in queries if q["image_id"] in keep]
    unknown = {q["image_id"] for q in queries} - {r["image_id"] for r in gallery}
    if unknown:
        raise SystemExit(
            f"queries reference image_ids missing from the gallery: {sorted(unknown)[:5]}"
        )
    return gallery, queries


def check_images(data_dir: Path, gallery: list[dict]) -> dict[str, str]:
    """Check every file exists and matches its recorded sha256; return {image_id: sha256}."""
    missing, mismatched, shas = [], [], {}
    for row in gallery:
        path = image_path(data_dir, row)
        if not path.exists():
            missing.append(str(path))
            continue
        actual = sha256_file(path)
        shas[row["image_id"]] = actual
        expected = row.get("sha256")
        if expected and expected != actual:
            mismatched.append(row["image_id"])
    if missing or mismatched:
        msg = [f"{len(missing)} image file(s) missing, {len(mismatched)} sha256 mismatch(es)"]
        msg += [f"  missing: {m}" for m in missing[:5]]
        msg += [f"  mismatch: {m}" for m in mismatched[:5]]
        raise SystemExit("\n".join(msg))
    return shas


# --------------------------------------------------------------------------------------
# Timing store (for the ETA)
# --------------------------------------------------------------------------------------
def load_timings(cache_dir: Path) -> dict[str, float]:
    path = cache_dir / "timings.json"
    if path.exists():
        try:
            return {**DEFAULT_SEC, **json.loads(path.read_text(encoding="utf-8"))}
        except (OSError, ValueError):
            pass
    return dict(DEFAULT_SEC)


def save_timing(cache_dir: Path, key: str, sec_per_item: float) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "timings.json"
    cur = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cur = {}
    cur[key] = round(sec_per_item, 4)
    path.write_text(json.dumps(cur, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------------------
# CLIP
# --------------------------------------------------------------------------------------
def l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def clip_cache_dir(cache_dir: Path) -> Path:
    return cache_dir / "clip" / CLIP_HF_ID.replace("/", "__")


class ClipEncoder:
    """OpenAI CLIP ViT-B/32 through transformers; vectors are returned L2-normalised."""

    size = 224

    def __init__(self, device: str):
        import torch
        from transformers import CLIPTokenizerFast

        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda is not available")
        self.torch = torch
        self.device = device
        self.tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_HF_ID)
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self._model = None

    @property
    def model(self):
        """Load the weights on first use, so a fully cached re-run never loads them."""
        if self._model is None:
            from transformers import CLIPModel

            t0 = time.perf_counter()
            self._model = CLIPModel.from_pretrained(CLIP_HF_ID).eval().to(self.device)
            print(f"[clip] model loaded in {time.perf_counter() - t0:.1f}s on {self.device}")
        return self._model

    def _prep(self, path: Path) -> np.ndarray:
        """Resize shorter side to 224 (bicubic), centre-crop, scale, normalise -> CHW float32."""
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = self.size / min(w, h)
            nw, nh = max(self.size, round(w * scale)), max(self.size, round(h * scale))
            im = im.resize((nw, nh), Image.BICUBIC)
            left, top = (nw - self.size) // 2, (nh - self.size) // 2
            im = im.crop((left, top, left + self.size, top + self.size))
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return ((arr - self.mean) / self.std).transpose(2, 0, 1)

    @staticmethod
    def _features(out):
        # transformers 5 may return a model-output object rather than a tensor
        return out if hasattr(out, "shape") else out.pooler_output

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        torch = self.torch
        batch = torch.from_numpy(np.stack([self._prep(p) for p in paths])).to(self.device)
        with torch.inference_mode():
            feats = self._features(self.model.get_image_features(pixel_values=batch))
        return l2_normalise(feats.float().cpu().numpy())

    def count_truncated(self, texts: list[str]) -> int:
        ids = self.tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
        return sum(len(i) > CONTEXT_LENGTH for i in ids)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=CONTEXT_LENGTH, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            feats = self._features(self.model.get_text_features(**enc))
        return l2_normalise(feats.float().cpu().numpy())


def run_clip(args, data_dir, gallery, queries, shas, cache_dir) -> dict:
    enc = ClipEncoder(args.device)
    ccache = clip_cache_dir(cache_dir)
    (ccache / "img").mkdir(parents=True, exist_ok=True)
    (ccache / "txt").mkdir(parents=True, exist_ok=True)

    # images, keyed by sha256
    vecs: dict[str, np.ndarray] = {}
    todo = []
    for row in gallery:
        f = ccache / "img" / f"{shas[row['image_id']]}.npy"
        if f.exists():
            vecs[row["image_id"]] = np.load(f)
        else:
            todo.append(row)
    print(
        f"[clip] images: {len(gallery) - len(todo)} cached, {len(todo)} to encode "
        f"(device {args.device})"
    )
    t1 = time.perf_counter()
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i : i + args.batch_size]
        out = enc.encode_images([image_path(data_dir, r) for r in chunk])
        for r, v in zip(chunk, out, strict=True):
            np.save(ccache / "img" / f"{shas[r['image_id']]}.npy", v)
            vecs[r["image_id"]] = v
        print(f"[clip]   {min(i + args.batch_size, len(todo))}/{len(todo)} images", flush=True)
    if todo:
        save_timing(cache_dir, "clip_img", (time.perf_counter() - t1) / len(todo))

    # captions, keyed by the text's sha256
    texts = [q["text"] for q in queries]
    tkeys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    tvecs: dict[str, np.ndarray] = {}
    ttodo = []
    for key, text in zip(tkeys, texts, strict=True):
        f = ccache / "txt" / f"{key}.npy"
        if f.exists():
            tvecs[key] = np.load(f)
        elif key not in {k for k, _ in ttodo}:
            ttodo.append((key, text))
    print(f"[clip] captions: {len(set(tkeys)) - len(ttodo)} cached, {len(ttodo)} to encode")
    t2 = time.perf_counter()
    bs = max(args.batch_size, 32)
    for i in range(0, len(ttodo), bs):
        chunk = ttodo[i : i + bs]
        out = enc.encode_texts([t for _, t in chunk])
        for (key, _), v in zip(chunk, out, strict=True):
            np.save(ccache / "txt" / f"{key}.npy", v)
            tvecs[key] = v
    if ttodo:
        save_timing(cache_dir, "clip_txt", (time.perf_counter() - t2) / len(ttodo))
    truncated = enc.count_truncated(texts)

    img = np.stack([vecs[r["image_id"]] for r in gallery]).astype(np.float32)
    txt = np.stack([tvecs[k] for k in tkeys]).astype(np.float32)
    import torch
    import transformers

    out_dir = data_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "clip_img.npy", img)
    np.save(out_dir / "clip_txt.npy", txt)
    meta = {
        "model_name": CLIP_MODEL_NAME,
        "pretrained": CLIP_PRETRAINED,
        "hf_checkpoint": CLIP_HF_ID,
        "dim": int(img.shape[1]),
        "normalised": "l2 (every row of clip_img.npy and clip_txt.npy has unit norm)",
        "image_ids": [r["image_id"] for r in gallery],
        "qids": [q["qid"] for q in queries],
        "library": "transformers",
        "library_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": args.device,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "context_length": CONTEXT_LENGTH,
        "n_captions": len(texts),
        "truncated_captions": int(truncated),
        "limit": args.limit,
        "image_sha256": {r["image_id"]: shas[r["image_id"]] for r in gallery},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[clip] wrote {img.shape} images, {txt.shape} captions, truncated={truncated}")
    return meta


# --------------------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------------------
def clahe_bgr(image: np.ndarray) -> np.ndarray:
    """Same CLAHE as the document scanner project: gray -> CLAHE(2.0, 8x8) -> 3-channel."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def ocr_engine_string(models: str) -> str:
    import paddleocr

    det, rec = OCR_MODELS[models]
    return f"paddleocr {paddleocr.__version__} {det}+{rec}"


def build_ocr(models: str, threads: int = 4):
    """Build the PaddleOCR pipeline.

    ``cpu_threads`` stays small and oneDNN is off on purpose: measured on this machine (12
    threads, shared with other jobs) 2-4 threads ran about 1.3-1.7 s/image on the mobile
    models, while 6-8 threads or oneDNN were 2-4x slower.
    """
    from paddleocr import PaddleOCR

    det, rec = OCR_MODELS[models]
    return PaddleOCR(
        text_detection_model_name=det,
        text_recognition_model_name=rec,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        cpu_threads=threads,
        enable_mkldnn=False,
    )


def ocr_one(engine, path: Path, preprocessing: str) -> list[dict]:
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise RuntimeError(f"cannot decode {path}")
    if preprocessing == "clahe":
        img = clahe_bgr(img)
    page = engine.predict(img)[0]
    texts, scores = page.get("rec_texts", []), page.get("rec_scores", [])
    return [
        {"text": t, "confidence": round(float(s), 4)}
        for t, s in zip(texts, scores, strict=True)
        if str(t).strip()
    ]


def run_ocr(args, data_dir, gallery, shas, cache_dir) -> None:
    ocr_dir = data_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    jsonl, sidecar = ocr_dir / "ocr.jsonl", ocr_dir / "ocr_meta.json"
    engine_name = ocr_engine_string(args.ocr_models)
    prep = args.ocr_preprocessing
    ocache = cache_dir / "ocr" / f"{args.ocr_models}_{prep}"
    ocache.mkdir(parents=True, exist_ok=True)

    # resume state: rows already in ocr.jsonl that were computed from the same image bytes
    # with the same engine and preprocessing
    side = {"image_sha256": {}}
    if sidecar.exists():
        with contextlib.suppress(ValueError):
            side = json.loads(sidecar.read_text(encoding="utf-8"))
    done: dict[str, dict] = {}
    if jsonl.exists():
        for row in read_jsonl(jsonl):
            iid = row["image_id"]
            if (
                row.get("engine") == engine_name
                and row.get("preprocessing") == prep
                and side.get("image_sha256", {}).get(iid) == shas.get(iid)
            ):
                done[iid] = row
    results = {iid: row for iid, row in done.items() if iid in shas}
    sha_map = {iid: shas[iid] for iid in results}

    def flush_sidecar() -> None:
        tmp = sidecar.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"engine": engine_name, "preprocessing": prep, "image_sha256": sha_map}, indent=1
            ),
            encoding="utf-8",
        )
        tmp.replace(sidecar)

    # rewrite jsonl with only the valid rows, then append as we go (crash-safe resume)
    with jsonl.open("w", encoding="utf-8") as fh:
        for row in results.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    flush_sidecar()

    todo = [r for r in gallery if r["image_id"] not in results]
    # same bytes already OCR'd under another id (or after jsonl was lost): reuse the cache
    engine = None
    print(f"[ocr] {len(results)} done, {len(todo)} to run ({engine_name}, preprocessing={prep})")
    t0, ran = time.perf_counter(), 0
    with jsonl.open("a", encoding="utf-8") as fh:
        for n, row in enumerate(todo, 1):
            iid, sha = row["image_id"], shas[row["image_id"]]
            cfile = ocache / f"{sha}.json"
            if cfile.exists():
                lines = json.loads(cfile.read_text(encoding="utf-8"))
            else:
                if engine is None:
                    engine = build_ocr(args.ocr_models, args.ocr_threads)
                lines = ocr_one(engine, image_path(data_dir, row), prep)
                cfile.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
                ran += 1
            out = {"image_id": iid, "lines": lines, "engine": engine_name, "preprocessing": prep}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            results[iid], sha_map[iid] = out, sha
            if n % 10 == 0 or n == len(todo):
                flush_sidecar()
                el = time.perf_counter() - t0
                print(f"[ocr]   {n}/{len(todo)} ({el:.0f}s elapsed)", flush=True)
    flush_sidecar()
    if ran:
        save_timing(cache_dir, "ocr", (time.perf_counter() - t0) / ran)

    # final pass: deterministic gallery order
    with jsonl.open("w", encoding="utf-8") as fh:
        for r in gallery:
            fh.write(json.dumps(results[r["image_id"]], ensure_ascii=False) + "\n")
    n_text = sum(1 for r in gallery if results[r["image_id"]]["lines"])
    print(f"[ocr] wrote {len(gallery)} rows ({n_text} images with at least one text line)")


# --------------------------------------------------------------------------------------
# Dry run and verify (torch-free)
# --------------------------------------------------------------------------------------
def fmt_dur(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def dry_run(args, data_dir, gallery, queries, shas, cache_dir) -> None:
    tm = load_timings(cache_dir)
    cc = clip_cache_dir(cache_dir)
    n_img_todo = sum(not (cc / "img" / f"{shas[r['image_id']]}.npy").exists() for r in gallery)
    n_txt_todo = len(
        {
            k
            for q in queries
            if not (
                cc / "txt" / f"{(k := hashlib.sha256(q['text'].encode()).hexdigest())}.npy"
            ).exists()
        }
    )
    oc = cache_dir / "ocr" / f"{args.ocr_models}_{args.ocr_preprocessing}"
    n_ocr_todo = sum(not (oc / f"{shas[r['image_id']]}.json").exists() for r in gallery)
    sources: dict[str, int] = {}
    for r in gallery:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    print(f"data dir        : {data_dir}")
    print(f"gallery         : {len(gallery)} images {sources}; all files present, sha256 match")
    print(f"queries         : {len(queries)}")
    print(f"device          : {args.device}   batch size: {args.batch_size}")
    print(f"OCR             : {args.ocr_models} models, preprocessing={args.ocr_preprocessing}")
    eta_clip = n_img_todo * tm["clip_img"] + n_txt_todo * tm["clip_txt"] + 40  # + model load
    eta_ocr = n_ocr_todo * tm["ocr"] + 20
    print(
        f"to compute      : clip images {n_img_todo}, clip captions {n_txt_todo}, ocr {n_ocr_todo}"
    )
    if args.only in ("clip", "all"):
        print(
            f"ETA clip        : ~{fmt_dur(eta_clip)} "
            f"({tm['clip_img']:.3f}s/img, {tm['clip_txt']:.3f}s/caption)"
        )
    if args.only in ("ocr", "all"):
        print(f"ETA ocr         : ~{fmt_dur(eta_ocr)} ({tm['ocr']:.2f}s/img)")
    print("dry run: no model loaded, nothing written")


def verify(args, data_dir) -> int:
    """Reload the outputs and check shapes, id sets, normalisation and NaNs. Needs numpy only."""
    gallery, queries = load_inputs(data_dir, args.limit)
    emb = data_dir / "embeddings"
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    meta = json.loads((emb / "meta.json").read_text(encoding="utf-8"))
    img = np.load(emb / "clip_img.npy")
    txt = np.load(emb / "clip_txt.npy")
    for name, arr in (("clip_img", img), ("clip_txt", txt)):
        check(arr.dtype == np.float32, f"{name} dtype {arr.dtype} != float32")
        check(arr.ndim == 2, f"{name} is not 2-D")
        check(bool(np.isfinite(arr).all()), f"{name} has NaN/inf")
        norms = np.linalg.norm(arr, axis=1)
        check(
            bool(np.allclose(norms, 1.0, atol=1e-4)),
            f"{name} rows not unit norm (min {norms.min():.5f}, max {norms.max():.5f})",
        )
    check(img.shape == (len(meta["image_ids"]), meta["dim"]), f"clip_img shape {img.shape}")
    check(txt.shape == (len(meta["qids"]), meta["dim"]), f"clip_txt shape {txt.shape}")
    check(set(meta["image_ids"]) == {r["image_id"] for r in gallery}, "meta image_ids != gallery")
    check(len(meta["image_ids"]) == len(set(meta["image_ids"])), "duplicate image_ids in meta")
    check(set(meta["qids"]) == {q["qid"] for q in queries}, "meta qids != queries")
    check(len(meta["qids"]) == len(set(meta["qids"])), "duplicate qids in meta")
    for key in (
        "model_name",
        "pretrained",
        "dim",
        "library",
        "library_version",
        "torch_version",
        "device",
        "created_utc",
        "image_sha256",
        "truncated_captions",
    ):
        check(key in meta, f"meta.json missing {key}")
    for r in gallery:
        recorded = meta.get("image_sha256", {}).get(r["image_id"])
        if r.get("sha256") and recorded != r["sha256"]:
            errors.append(f"sha256 for {r['image_id']} differs from gallery.jsonl")
            break

    ocr_rows = read_jsonl(data_dir / "ocr" / "ocr.jsonl")
    ocr_ids = [r["image_id"] for r in ocr_rows]
    check(set(ocr_ids) == {r["image_id"] for r in gallery}, "ocr.jsonl ids != gallery")
    check(len(ocr_ids) == len(set(ocr_ids)), "duplicate image_ids in ocr.jsonl")
    for row in ocr_rows:
        ok = isinstance(row.get("lines"), list) and all(
            isinstance(ln.get("text"), str) and 0.0 <= float(ln.get("confidence", -1)) <= 1.0
            for ln in row["lines"]
        )
        if not (ok and row.get("engine") and row.get("preprocessing")):
            errors.append(f"malformed OCR row for {row.get('image_id')}")
            break
    n_text = sum(1 for r in ocr_rows if r["lines"])
    if errors:
        print("VERIFY FAILED")
        for e in errors:
            print(" -", e)
        return 1
    print(
        f"VERIFY OK: clip_img {img.shape}, clip_txt {txt.shape}, {len(ocr_rows)} OCR rows "
        f"({n_text} with text), truncated captions "
        f"{meta['truncated_captions']}/{meta['n_captions']}, "
        f"engine '{ocr_rows[0]['engine'] if ocr_rows else '-'}'"
    )
    return 0


# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--only", choices=["clip", "ocr", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N gallery images only")
    ap.add_argument("--dry-run", action="store_true", help="validate inputs, print plan and ETA")
    ap.add_argument("--verify", action="store_true", help="check written outputs (numpy only)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ocr-preprocessing", choices=OCR_PREPROCESSING, default="none")
    ap.add_argument("--ocr-models", choices=sorted(OCR_MODELS), default=DEFAULT_OCR_MODELS)
    ap.add_argument("--ocr-threads", type=int, default=4, help="PaddleOCR CPU threads")
    ap.add_argument("--cache-dir", type=Path, default=None, help="default: <data-dir>/.cache")
    args = ap.parse_args(argv)

    data_dir = args.data_dir.resolve()
    cache_dir = (args.cache_dir or data_dir / ".cache").resolve()
    if args.verify:
        return verify(args, data_dir)

    gallery, queries = load_inputs(data_dir, args.limit)
    shas = check_images(data_dir, gallery)
    if args.dry_run:
        dry_run(args, data_dir, gallery, queries, shas, cache_dir)
        return 0

    if args.only in ("clip", "all"):
        run_clip(args, data_dir, gallery, queries, shas, cache_dir)
    if args.only in ("ocr", "all"):
        run_ocr(args, data_dir, gallery, shas, cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
