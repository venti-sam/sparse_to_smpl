"""
BONES-SEED (soma_uniform) -> v2 training files.

    python3 bones_pipeline.py download
    python3 bones_pipeline.py convert
    python3 bones_pipeline.py index        # needs pandas + pyarrow (one-off)

download: resumable, re-resolves the signed URL on every attempt, sha256-checked.
convert:  reads the local .tar.gz as a stream, skips mirrored clips (names ending
          _M), converts the rest in parallel with convert_bvh_v2 (SMPL body) and
          writes <dst>/BONES_<name>.pt. Safe to re-run: finished clips are kept.
index:    writes <dst>/index.json (category, body position, actor, session, ...)
          per clip, for filtering / sampling at training time.

Needs a Hugging Face read token in HF_TOKEN or ~/.cache/huggingface/token, after
accepting the BONES-SEED licence (research use only: do not commit or share the
data or the converted files).
"""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

REPO = "bones-studio/seed"
HF = "https://huggingface.co"
ARCHIVE = "soma_uniform.tar.gz"
DATA = "../support_data/bones_seed"
DST = "../support_data/bones_uniform_v2"
BASE_SKEL = f"{DATA}/soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh"
EXPECTED_NON_MIRROR = 71132  # from the dataset card


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ─────────────────────────── download ───────────────────────────


def _token():
    t = os.environ.get("HF_TOKEN")
    if not t:
        with open(os.path.expanduser("~/.cache/huggingface/token")) as f:
            t = f.read()
    return t.strip()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def signed_url(name):
    """First hop only: the token is sent to huggingface.co, never to the CDN.
    Returns None when the file is served directly (small non-LFS files)."""
    req = urllib.request.Request(f"{HF}/datasets/{REPO}/resolve/main/{name}",
                                 headers={"Authorization": f"Bearer {_token()}"})
    try:
        urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return e.headers["Location"]
        raise
    return None


def fetch_direct(name, dest):
    req = urllib.request.Request(f"{HF}/datasets/{REPO}/resolve/main/{name}",
                                 headers={"Authorization": f"Bearer {_token()}"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while b := r.read(1 << 20):
            f.write(b)


def remote_info(name):
    """(size, sha256 or None) of a file in the repo (non-LFS files have no checksum)."""
    folder = os.path.dirname(name)
    req = urllib.request.Request(f"{HF}/api/datasets/{REPO}/tree/main/{folder}?expand=true",
                                 headers={"Authorization": f"Bearer {_token()}"})
    for e in json.load(urllib.request.urlopen(req, timeout=60)):
        if e["path"] == name:
            return e["size"], (e.get("lfs") or {}).get("oid")
    raise FileNotFoundError(name)


def sha256(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(args):
    os.makedirs(args.out, exist_ok=True)
    final = os.path.join(args.out, args.file)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    part = final + ".part"
    size, oid = remote_info(args.file)
    log(f"{args.file}: {size / 1e9:.2f} GB")

    def size_now(p):
        return os.path.getsize(p) if os.path.exists(p) else 0

    if size_now(final) != size:
        attempts = 0
        while size_now(part) < size:
            attempts += 1
            if attempts > args.max_attempts:
                raise RuntimeError("too many failed attempts; re-run to resume")
            log(f"attempt {attempts}: resuming at {size_now(part) / 1e9:.2f} GB")
            url = signed_url(args.file)
            if url is None:  # small file, no CDN
                fetch_direct(args.file, part)
                continue
            proc = subprocess.Popen(["curl", "-s", "-S", "--fail", "-C", "-", "-o", part, url])
            t0, b0 = time.time(), size_now(part)
            while proc.poll() is None:
                time.sleep(60)
                b = size_now(part)
                rate = (b - b0) / max(time.time() - t0, 1e-9)
                log(f"  {b / 1e9:6.2f} / {size / 1e9:.2f} GB ({100 * b / size:4.1f}%)  {rate / 1e6:5.1f} MB/s  "
                    f"ETA {(size - b) / max(rate, 1) / 60:5.1f} min")
            if proc.returncode != 0:
                log(f"  curl exited {proc.returncode}; retrying in 10 s")
                time.sleep(10)
        os.replace(part, final)
    if oid is None:
        log("no checksum published for this file:", final)
        return
    log("verifying sha256 ...")
    if sha256(final) != oid:
        raise RuntimeError(f"checksum mismatch for {final}: delete it and re-run")
    log("sha256 OK:", final)


# ─────────────────────────── convert ───────────────────────────

_CTX = None


def _init(ctx):
    global _CTX
    import torch
    torch.set_num_threads(1)
    _CTX = ctx


def _work(stem, data, extra, out_name):
    from convert_bvh_v2 import process_clip
    r = process_clip(_CTX, data, stem, extra, out_name)
    r["stem"] = stem
    return r


def convert_archive(args):
    from convert_bvh_v2 import make_context, summarize
    ns = argparse.Namespace(skeleton="soma", ref_file=args.ref_file, ref_frame=0, dst=args.dst,
                            prefix="BONES", body="smpl", body_scale=1.0)
    ctx = make_context(ns)
    os.makedirs(args.dst, exist_ok=True)
    existing = set(os.listdir(args.dst))

    results, fails, seen = [], [], set()
    n_members = n_mirror = n_have = n_sub = 0
    t0 = time.time()
    window = 4 * args.workers

    def collect(done):
        for fut in done:
            r = fut.result()
            results.append(r)
            if not r["ok"]:
                fails.append((r["stem"], r["msg"]))
        if len(results) // 2000 != (len(results) - len(done)) // 2000:
            rate = len(results) / (time.time() - t0)
            log(f"converted {len(results)} (+{n_have} already done), mirrors skipped {n_mirror}, "
                f"{len(fails)} failed, {rate:.0f} clips/s, ~{(EXPECTED_NON_MIRROR - len(results) - n_have) / max(rate, 1e-9) / 60:.0f} min left")

    pending = set()
    with tarfile.open(args.archive, "r|gz") as tar, \
            ProcessPoolExecutor(args.workers, initializer=_init, initargs=(ctx,)) as ex:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".bvh")):
                continue
            n_members += 1
            parts = m.name.split("/")
            stem, session = parts[-1][:-4], parts[-2]
            if stem.endswith("_M") and not args.include_mirrors:
                n_mirror += 1
                continue
            name = f"BONES_{stem}.pt"
            if stem in seen:  # same name in two sessions
                name = f"BONES_{session}_{stem}.pt"
            seen.add(stem)
            if name in existing:
                n_have += 1
                continue
            pending.add(ex.submit(_work, stem, tar.extractfile(m).read(), {"session": session}, name))
            n_sub += 1
            if len(pending) >= window:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)
            if args.limit and n_sub >= args.limit:
                break
        while pending:
            done, pending = wait(pending)
            collect(done)

    log(f"archive members: {n_members} .bvh | mirrors skipped: {n_mirror} | already converted: {n_have} | "
        f"converted now: {sum(r['ok'] for r in results)} | failed: {len(fails)}")
    summarize(results)
    for stem, msg in fails[:20]:
        print("  !", stem, msg)
    json.dump({"members": n_members, "mirrors_skipped": n_mirror, "already_done": n_have,
               "converted": sum(r["ok"] for r in results), "failed": fails},
              open(os.path.join(args.dst, "convert_report.json"), "w"), indent=1)


# ─────────────────────────── index ───────────────────────────

INDEX_COLS = ["package", "category", "content_body_position", "content_type_of_movement",
              "content_uniform_style", "content_props", "content_horizontal_move", "content_vertical_move",
              "actor_uid", "actor_height_cm", "take_date", "move_duration_frames", "content_short_description"]


def build_index(args):
    import pandas as pd
    df = pd.read_parquet(args.metadata)
    have = {f[len("BONES_"):-3] for f in os.listdir(args.dst) if f.endswith(".pt")}
    sub = df[df["filename"].isin(have)]
    index = {r["filename"]: {c: (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c], "item") else r[c]))
                             for c in INDEX_COLS} for _, r in sub.iterrows()}
    json.dump(index, open(os.path.join(args.dst, "index.json"), "w"))
    log(f"index: {len(index)} of {len(have)} converted clips found in metadata "
        f"(non-mirror clips in metadata: {int((~df['is_mirror']).sum())})")
    print("body position:", sub["content_body_position"].value_counts().to_dict())
    print("package:", sub["package"].value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("download")
    d.add_argument("--file", default=ARCHIVE)
    d.add_argument("--out", default=DATA)
    d.add_argument("--max-attempts", type=int, default=100)
    c = sp.add_parser("convert")
    c.add_argument("--archive", default=f"{DATA}/{ARCHIVE}")
    c.add_argument("--dst", default=DST)
    c.add_argument("--ref-file", default=BASE_SKEL)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--limit", type=int, default=None, help="stop after N clips (testing)")
    c.add_argument("--include-mirrors", action="store_true")
    i = sp.add_parser("index")
    i.add_argument("--dst", default=DST)
    i.add_argument("--metadata", default=f"{DATA}/metadata/seed_metadata_v004.parquet")
    args = ap.parse_args()
    {"download": download, "convert": convert_archive, "index": build_index}[args.cmd](args)
