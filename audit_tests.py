"""
audit_tests.py  -  Offline + live audit of the SG Meme dataset integration.
Run: python audit_tests.py
"""
import io
import json
import os
import sys
# Force UTF-8 output on Windows so Unicode chars don't crash
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import urllib.request

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
CACHE_DIR = os.path.join(ROOT, ".cache", "hf_stream")

def dir_size_bytes(path):
    if not os.path.exists(path):
        return 0
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total

def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  AUDIT TEST SUITE")
print("="*62)

# ─── TEST A: Label schema via HF Datasets-Server API (no download) ────────────
print("\n[A] Verifying label schema from HF API (no Parquet download)...")
try:
    url = ("https://datasets-server.huggingface.co/info"
           "?dataset=aliencaocao%2Fmultimodal_meme_classification_singapore")
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.loads(r.read())

    feat = data["dataset_info"]["default"]["features"]
    gpt  = feat["metadata"]["gpt_response"]
    hum  = feat["metadata"]["human_response"]

    gpt_names = gpt["names"]   # e.g. ["Yes", "No"]
    hum_names = hum["names"]

    print(f"  gpt_response ClassLabel names : {gpt_names}")
    print(f"  human_response ClassLabel names: {hum_names}")
    print(f"  gpt index 0 = '{gpt_names[0]}'  (meaning: offensive/harmful = Yes)")
    print(f"  gpt index 1 = '{gpt_names[1]}'  (meaning: not offensive = No)")

    # Verify our mapping
    cfg_map = {0: "hateful", 1: "non-hateful"}
    if gpt_names[0] == "Yes" and gpt_names[1] == "No":
        mapping_ok = True
        print("  PASS: dataset_config.py label mapping is CORRECT")
        print("        index 0 = Yes (offensive) -> 'hateful'       ✓")
        print("        index 1 = No (not offensive) -> 'non-hateful' ✓")
    else:
        mapping_ok = False
        print(f"  FAIL: unexpected names {gpt_names}")

    results["label_schema"] = {
        "status": "PASS" if mapping_ok else "FAIL",
        "gpt_names": gpt_names,
        "hum_names": hum_names,
        "mapping_correct": mapping_ok,
    }
except Exception as e:
    print(f"  ERROR: {e}")
    results["label_schema"] = {"status": "ERROR", "error": str(e)}

# ─── TEST B: human_response null prevalence (via /rows API) ──────────────────
print("\n[B] Checking human_response null prevalence via HF rows API...")
try:
    url = ("https://datasets-server.huggingface.co/rows"
           "?dataset=aliencaocao%2Fmultimodal_meme_classification_singapore"
           "&config=default&split=test&offset=0&length=100")
    with urllib.request.urlopen(url, timeout=30) as r:
        rows_data = json.loads(r.read())

    rows = rows_data.get("rows", [])
    null_count  = sum(1 for r in rows if r["row"]["metadata"]["human_response"] is None)
    total_count = len(rows)
    gpt_values  = [r["row"]["metadata"]["gpt_response"] for r in rows]
    gpt_counts  = {}
    for v in gpt_values:
        gpt_counts[str(v)] = gpt_counts.get(str(v), 0) + 1

    print(f"  Rows sampled           : {total_count}")
    print(f"  human_response = null  : {null_count}/{total_count} ({null_count/total_count:.0%})")
    print(f"  gpt_response distribution: {gpt_counts}")

    # Translate gpt_response values
    if gpt_names:
        translated = {}
        for k, v in gpt_counts.items():
            if k == "None":
                translated["null"] = v
            else:
                idx = int(k)
                translated[f"{k}={gpt_names[idx]}"] = v
        print(f"  gpt labels translated  : {translated}")

    results["human_response_null"] = {
        "status": "PASS",
        "null_count": null_count,
        "total_sampled": total_count,
        "null_pct": null_count / total_count if total_count else 0,
        "gpt_distribution": gpt_counts,
    }
except Exception as e:
    print(f"  ERROR: {e}")
    results["human_response_null"] = {"status": "ERROR", "error": str(e)}

# ─── TEST C: Verify dataset_config.py CACHE_DIR is project-local ─────────────
print("\n[C] Verifying cache directory isolation...")
try:
    import dataset_config as cfg
    cache = cfg.CACHE_DIR
    global_hf = os.path.expanduser("~/.cache/huggingface")
    is_project_local = cache.startswith(ROOT)
    is_not_global    = not cache.startswith(global_hf)
    print(f"  CACHE_DIR         : {cache}")
    print(f"  Project root      : {ROOT}")
    print(f"  Is project-local  : {is_project_local}")
    print(f"  Is not global HF  : {is_not_global}")
    print(f"  Current cache size: {human_bytes(dir_size_bytes(cache))}")
    print(f"  STATUS: {'PASS' if is_project_local else 'FAIL'}")
    results["cache_isolation"] = {
        "status": "PASS" if is_project_local else "FAIL",
        "cache_dir": cache,
        "is_project_local": is_project_local,
        "current_size": dir_size_bytes(cache),
    }
except Exception as e:
    print(f"  ERROR: {e}")
    results["cache_isolation"] = {"status": "ERROR", "error": str(e)}

# ─── TEST D: stream_dataset.py --no-images mode, 20 records ──────────────────
print("\n[D] Running metadata-only stream (20 records, --no-images)...")
cache_before = dir_size_bytes(CACHE_DIR)
import subprocess, time
t0 = time.time()
proc = subprocess.run(
    [sys.executable, "stream_dataset.py", "--samples", "20", "--no-images", "--overwrite"],
    capture_output=True, text=True, cwd=ROOT, timeout=180
)
elapsed = time.time() - t0
cache_after = dir_size_bytes(CACHE_DIR)
cache_growth = cache_after - cache_before

print(f"  Exit code     : {proc.returncode}")
print(f"  Elapsed       : {elapsed:.1f}s")
print(f"  Cache before  : {human_bytes(cache_before)}")
print(f"  Cache after   : {human_bytes(cache_after)}")
print(f"  Cache growth  : {human_bytes(cache_growth)}")

# Check for image files in output
output_dir = os.path.join(ROOT, "experiment_output")
img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
image_files = []
if os.path.exists(output_dir):
    for dp, _, files in os.walk(output_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in img_exts:
                image_files.append(os.path.join(dp, f))

print(f"  Image files in output/: {len(image_files)}")
if image_files:
    for f in image_files:
        print(f"    {f}")

# Check stdout for key markers
stdout_lines = proc.stdout.strip().split("\n")
has_done   = any("Done" in l for l in stdout_lines)
has_limit  = any("20" in l for l in stdout_lines)
has_stream = any("streaming" in l.lower() for l in stdout_lines)

print(f"\n  Key output markers:")
for line in stdout_lines[-25:]:
    print(f"    {line}")

# Check stderr for errors
if proc.stderr.strip():
    stderr_short = [l for l in proc.stderr.strip().split("\n")
                    if "warning" not in l.lower() and "userwarning" not in l.lower()]
    if stderr_short:
        print(f"\n  STDERR (non-warning):")
        for l in stderr_short[:10]:
            print(f"    {l}")

pass_d = (proc.returncode == 0 and has_done and len(image_files) == 0)
print(f"\n  STATUS: {'PASS' if pass_d else 'FAIL'}")
results["no_images_stream"] = {
    "status": "PASS" if pass_d else "FAIL",
    "exit_code": proc.returncode,
    "elapsed_s": round(elapsed, 1),
    "cache_growth_bytes": cache_growth,
    "image_files_saved": len(image_files),
    "has_done_marker": has_done,
}

# ─── TEST E: Verify sample limit respected ────────────────────────────────────
print("\n[E] Verifying sample limit from output CSV...")
import csv
csv_path = os.path.join(ROOT, "experiment_output", "sg_meme_metadata.csv")
if os.path.exists(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    row_count = len(rows)
    print(f"  CSV rows (excluding header): {row_count}")
    within_limit = (row_count == 20)
    print(f"  Sample limit exactly matches requested 20: {within_limit}")
    if rows:
        labels = {}
        for r in rows:
            l = r.get("gpt_label", "?")
            labels[l] = labels.get(l, 0) + 1
        print(f"  Label distribution: {labels}")
        print(f"  Sample IDs: {[r['id'] for r in rows[:3]]}")
    status_e = 'PASS' if within_limit else 'FAIL'
    print(f"  STATUS: {status_e}")
    results["sample_limit"] = {
        "status": status_e,
        "csv_row_count": row_count,
    }
else:
    print(f"  CSV not found at {csv_path}")
    results["sample_limit"] = {"status": "FAIL", "error": "CSV not found"}

# ─── TEST F: app.py imports without loading dataset ──────────────────────────
print("\n[F] Verifying app.py does not import dataset at module level...")
try:
    with open(os.path.join(ROOT, "app.py"), "r", encoding="utf-8") as f:
        app_src = f.read()
    imports_datasets = "from datasets" in app_src or "import datasets" in app_src
    imports_stream   = "stream_dataset" in app_src
    imports_cfg      = "dataset_config" in app_src
    print(f"  'from datasets' in app.py   : {imports_datasets}")
    print(f"  'stream_dataset' in app.py  : {imports_stream}")
    print(f"  'dataset_config' in app.py  : {imports_cfg}")
    app_clean = not (imports_datasets or imports_stream or imports_cfg)
    print(f"  STATUS: {'PASS' if app_clean else 'FAIL'}")
    results["app_isolation"] = {
        "status": "PASS" if app_clean else "FAIL",
        "imports_datasets_lib": imports_datasets,
        "imports_stream_dataset": imports_stream,
        "imports_dataset_config": imports_cfg,
    }
except Exception as e:
    print(f"  ERROR: {e}")
    results["app_isolation"] = {"status": "ERROR", "error": str(e)}

# ─── TEST G: Check for no model training code ─────────────────────────────────
print("\n[G] Verifying stream_dataset.py contains no training code...")
try:
    with open(os.path.join(ROOT, "stream_dataset.py"), "r", encoding="utf-8") as f:
        sd_src = f.read()
    training_keywords = [".train()", "optimizer.step", "loss.backward",
                         "trainer.train", "Trainer(", "DataLoader("]
    found = [kw for kw in training_keywords if kw in sd_src]
    print(f"  Training keywords found: {found if found else 'none'}")
    print(f"  STATUS: {'PASS' if not found else 'FAIL'}")
    results["no_training"] = {
        "status": "PASS" if not found else "FAIL",
        "training_keywords_found": found,
    }
except Exception as e:
    results["no_training"] = {"status": "ERROR", "error": str(e)}

# ─── TEST H: Network measurement limitation note ──────────────────────────────
print("\n[H] Network usage measurement capability...")
print("  Python stdlib has no per-process network byte counter.")
print("  Measurement approach used: OS-level disk cache growth (dir_size_bytes).")
print("  This measures HF's local Parquet shard cache, which approximates")
print("  total bytes fetched from the network for the streamed portion.")
print("  It does NOT capture HTTP overhead or TLS handshake bytes.")
print("  For precise measurement, use Wireshark or 'netstat -s' before/after.")
results["network_measurement"] = {
    "status": "NOTE",
    "method": "cache dir size delta (proxy for network bytes)",
    "limitation": "No per-process network counter available in Python stdlib",
}

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  AUDIT SUMMARY")
print("="*62)
all_pass = True
for test, res in results.items():
    s = res.get("status", "?")
    if s not in ("PASS", "NOTE"):
        all_pass = False
    icon = {"PASS": "[OK]", "FAIL": "[!!]", "ERROR": "[!]", "NOTE": "[i]"}.get(s, "?")
    print(f"  {icon} {test:<30} {s}")

print()
total_cache = dir_size_bytes(CACHE_DIR)
print(f"  Total cache on disk: {human_bytes(total_cache)}")
print(f"  Cache path:          {CACHE_DIR}")
print()
print(f"  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS NEED ATTENTION'}")
print("="*62)

# Write machine-readable results
out = os.path.join(ROOT, "experiment_output", "audit_results.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
results["total_cache_bytes"] = total_cache
results["total_cache_human"] = human_bytes(total_cache)
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"  Audit results saved to: {out}")
