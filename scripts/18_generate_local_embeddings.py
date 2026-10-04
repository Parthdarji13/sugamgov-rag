"""
scripts/18_generate_local_embeddings.py
=========================================
Step 14: Resumable Local Embedding Generation for SugamGov AI
with Live Terminal Progress Dashboard.

Populates `scheme_chunks.embedding_local vector(384)` using
intfloat/multilingual-e5-small.

SAFETY GUARANTEES:
  - Only processes rows WHERE embedding_local IS NULL
  - Never touches the `embedding` (Gemini, 768-dim) column
  - Never deletes or overwrites existing rows
  - Commits in configurable batches — crash-safe and resumable
  - Dry-run mode: generates embeddings but skips DB write
  - Live terminal dashboard updating ~1s in-place via ANSI control

Usage:
  # Test the live terminal display (mock mode, no model or DB writes):
  python scripts/18_generate_local_embeddings.py --test-display

  # Dry-run (10 chunks, actual model, no DB write):
  python scripts/18_generate_local_embeddings.py --dry-run --limit 10

  # Normal run (all NULL rows):
  python scripts/18_generate_local_embeddings.py

  # Limited run (e.g. first 500 chunks):
  python scripts/18_generate_local_embeddings.py --limit 500

  # Custom batch size & refresh interval:
  python scripts/18_generate_local_embeddings.py --batch-size 64 --refresh-interval 1.0
"""

import os
import sys
import time
import math
import argparse
import signal
import threading
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import numpy as np

# ── UTF-8 stdout on Windows ───────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DB_URL = os.getenv("DATABASE_URL")

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME        = "intfloat/multilingual-e5-small"
LOCAL_DIM         = 384
DEFAULT_BATCH     = 64
DEFAULT_REFRESH   = 1.0

# ── Console initialization (Windows VT100 / ANSI escape support) ─────────────
def init_console():
    if os.name == "nt":
        # Calling os.system("") on Windows activates ANSI escape processing
        os.system("")

# ── Graceful interrupt handling ───────────────────────────────────────────────
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    _interrupted = True

signal.signal(signal.SIGINT, _handle_sigint)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_time(secs):
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

def render_bar(pct: float, width: int = 28) -> str:
    """Render a visual progress bar e.g. [████████████████░░░░░░░░░░░░] 51.52%"""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * (pct / 100.0)))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:5.2f}%"

def get_counts(conn):
    total = conn.execute(text(
        "SELECT count(id) FROM scheme_chunks"
    )).scalar()
    gem_pop = conn.execute(text(
        "SELECT count(id) FROM scheme_chunks WHERE embedding IS NOT NULL"
    )).scalar()
    loc_pop = conn.execute(text(
        "SELECT count(id) FROM scheme_chunks WHERE embedding_local IS NOT NULL"
    )).scalar()
    loc_null = total - loc_pop
    gem_null = total - gem_pop
    return {
        "total": total,
        "gem_pop": gem_pop,
        "gem_null": gem_null,
        "loc_pop": loc_pop,
        "loc_null": loc_null,
    }

def vec_to_pg(vec: np.ndarray) -> str:
    """Convert numpy float32 array to PostgreSQL vector literal string."""
    return "[" + ",".join(f"{v:.8f}" for v in vec.tolist()) + "]"

# ── Live Terminal Progress Dashboard ──────────────────────────────────────────
class LiveProgressDashboard:
    """
    Live terminal progress dashboard refreshing ~1s in-place using ANSI control.
    Accurately reflects only successfully committed embeddings in PostgreSQL.
    """
    def __init__(
        self,
        total_chunks: int,
        initial_completed: int,
        total_batches: int,
        historical_batches: int,
        refresh_interval: float = DEFAULT_REFRESH,
        is_dry_run: bool = False,
    ):
        self.total_chunks = total_chunks
        self.initial_completed = initial_completed
        self.total_batches = total_batches
        self.historical_batches = historical_batches
        self.refresh_interval = max(0.2, refresh_interval)
        self.is_dry_run = is_dry_run

        self.session_completed = 0
        self.current_batch = historical_batches
        self.errors_count = 0
        self.is_running = False
        self.start_time = None
        self.lock = threading.Lock()
        self.thread = None
        self.is_tty = sys.stdout.isatty()
        self.lines_rendered = 0
        self.last_non_tty_log_time = 0.0

    def start(self):
        self.start_time = time.time()
        self.is_running = True
        self.render()
        self.thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self.thread.start()

    def update_batch(self, chunks_committed: int, batch_index: int):
        with self.lock:
            self.session_completed += chunks_committed
            self.current_batch = self.historical_batches + batch_index
        self.render()

    def add_error(self, count: int = 1):
        with self.lock:
            self.errors_count += count
        self.render()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        # Final render before stopping
        self.render()

    def _refresh_loop(self):
        while self.is_running:
            time.sleep(self.refresh_interval)
            if not self.is_running:
                break
            self.render()

    def get_stats(self):
        with self.lock:
            elapsed = time.time() - self.start_time if self.start_time else 0.0
            completed = self.initial_completed + self.session_completed
            remaining = max(0, self.total_chunks - completed)
            pct = (completed / self.total_chunks * 100.0) if self.total_chunks > 0 else 100.0

            # Speed based on successfully committed chunks in current session
            speed = (self.session_completed / elapsed) if elapsed > 0 and self.session_completed > 0 else 0.0
            eta = (remaining / speed) if speed > 0 else 0.0

            return {
                "total": self.total_chunks,
                "completed": completed,
                "remaining": remaining,
                "pct": pct,
                "speed": speed,
                "elapsed": elapsed,
                "eta": eta,
                "batch": self.current_batch,
                "total_batches": self.total_batches,
                "errors": self.errors_count,
            }

    def render(self):
        stats = self.get_stats()
        now = time.time()

        if not self.is_tty:
            # Non-interactive / log file mode: throttle to avoid cluttering log
            if now - self.last_non_tty_log_time >= 10.0 or stats["completed"] == self.total_chunks:
                self.last_non_tty_log_time = now
                speed_str = f"{stats['speed']:.1f} chunks/sec" if stats["speed"] > 0 else "-- chunks/sec"
                eta_str = fmt_time(stats["eta"]) if stats["speed"] > 0 else "--"
                print(
                    f"  [Progress] {stats['completed']:,}/{stats['total']:,} ({stats['pct']:5.2f}%) | "
                    f"Batch {stats['batch']}/{stats['total_batches']} | "
                    f"Speed: {speed_str} | "
                    f"Elapsed: {fmt_time(stats['elapsed'])} | "
                    f"ETA: {eta_str} | "
                    f"Errors: {stats['errors']}",
                    flush=True
                )
            return

        # Interactive terminal: render in-place using ANSI control
        bar_str = render_bar(stats["pct"], width=28)
        speed_str = f"{stats['speed']:.1f} chunks/sec" if stats["speed"] > 0 else "-- chunks/sec"
        eta_str = fmt_time(stats["eta"]) if stats["speed"] > 0 else "estimating..."

        lines = [
            "Local Embedding Generation",
            "==========================",
            f"Total chunks       : {stats['total']:,}",
            f"Completed          : {stats['completed']:,}",
            f"Remaining          : {stats['remaining']:,}",
            f"Progress           : {bar_str}",
            f"Speed              : {speed_str}",
            f"Elapsed            : {fmt_time(stats['elapsed'])}",
            f"ETA                : {eta_str}",
            f"Batch              : {stats['batch']}/{stats['total_batches']}",
            f"Errors             : {stats['errors']}",
        ]

        output = []
        if self.lines_rendered > 0:
            # Move cursor up by previously rendered lines
            output.append(f"\033[{self.lines_rendered}A")

        for line in lines:
            # Clear line to end and write content
            output.append(f"\r\033[2K{line}\n")

        sys.stdout.write("".join(output))
        sys.stdout.flush()
        self.lines_rendered = len(lines)

# ── Mock Display Test (Safety validation without model or DB) ─────────────────
def run_mock_display(refresh_interval: float = 1.0):
    """Safely test and demonstrate the live terminal progress dashboard."""
    init_console()
    print("=" * 68)
    print("  SugamGov AI — Progress Dashboard Mock Validation Test")
    print("  (Simulating live progress across 5 batches — no DB writes)")
    print("=" * 68)
    print()

    total_chunks = 20497
    initial_completed = 10560
    batch_size = 64
    historical_batches = math.floor(initial_completed / batch_size)
    total_batches = math.ceil(total_chunks / batch_size)

    dashboard = LiveProgressDashboard(
        total_chunks=total_chunks,
        initial_completed=initial_completed,
        total_batches=total_batches,
        historical_batches=historical_batches,
        refresh_interval=refresh_interval,
        is_dry_run=True,
    )
    dashboard.start()

    try:
        # Simulate 5 batches with 1 second between updates
        for b in range(1, 6):
            time.sleep(1.0)
            dashboard.update_batch(chunks_committed=64, batch_index=b)
    finally:
        dashboard.stop()
        stats = dashboard.get_stats()

    print("\n" + "=" * 26)
    print("LOCAL EMBEDDING COMPLETE")
    print("=" * 26)
    print(f"Total:         {stats['total']:,}")
    print(f"Completed:     {stats['completed']:,}")
    print(f"Remaining:     {stats['remaining']:,}")
    print(f"Duration:      {fmt_time(stats['elapsed'])}")
    avg_speed_str = f"{stats['speed']:.1f} chunks/sec" if stats["speed"] > 0 else "N/A"
    print(f"Average speed: {avg_speed_str}")
    print(f"Errors:        {stats['errors']}")
    print("=" * 26)
    print("\n[SUCCESS] Mock display test completed cleanly.")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Local embedding generation for SugamGov AI")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate embeddings but skip DB writes")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of chunks to process (default: all NULL)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help=f"Encoding batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--refresh-interval", type=float, default=DEFAULT_REFRESH,
                        help=f"Dashboard refresh interval in seconds (default: {DEFAULT_REFRESH})")
    parser.add_argument("--test-display", action="store_true",
                        help="Run mock progress display test without model or DB writes")
    args = parser.parse_args()

    # Handle mock display test mode
    if args.test_display:
        run_mock_display(refresh_interval=args.refresh_interval)
        return

    init_console()

    print("=" * 68)
    print("  SugamGov AI — Step 14: Local Embedding Generation")
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Dimension:        {LOCAL_DIM}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Refresh interval: {args.refresh_interval}s")
    print(f"  Mode:             {'DRY-RUN (no DB writes)' if args.dry_run else 'LIVE'}")
    if args.limit:
        print(f"  Limit:            {args.limit} chunks")
    print("=" * 68)

    # ── 1. Connect & verify schema ────────────────────────────────────────────
    engine = create_engine(DB_URL)
    print("\n[1/5] Pre-flight checks...")

    with engine.connect() as conn:
        col = conn.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='scheme_chunks' AND column_name='embedding_local'"
        )).fetchone()
        if not col:
            print("  [ERROR] embedding_local column does not exist.")
            print("  Run Step 13 (validate_local_embedding_storage.py) first.")
            sys.exit(1)

        loc_dim = conn.execute(text(
            "SELECT vector_dims(embedding_local) FROM scheme_chunks "
            "WHERE embedding_local IS NOT NULL LIMIT 1"
        )).scalar()
        if loc_dim is None:
            attr = conn.execute(text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid='scheme_chunks'::regclass AND attname='embedding_local'"
            )).scalar()
            loc_dim = (attr - 4) if attr and attr > 4 else "unknown"

        c = get_counts(conn)

    print(f"  Total chunks:           {c['total']:,}")
    print(f"  Gemini populated:       {c['gem_pop']:,}")
    print(f"  Gemini NULL:            {c['gem_null']:,}")
    print(f"  embedding_local pop:    {c['loc_pop']:,}")
    print(f"  embedding_local NULL:   {c['loc_null']:,}")
    print(f"  embedding_local dim:    {loc_dim}")

    if c['total'] != 20497:
        print(f"  [WARN] Expected 20,497 total chunks, got {c['total']}.")
    if c['gem_pop'] != 1260:
        print(f"  [WARN] Expected 1,260 Gemini embeddings, got {c['gem_pop']}.")

    remaining = c['loc_null']
    if args.limit:
        remaining = min(remaining, args.limit)

    if remaining == 0:
        print("\n  [OK] All embedding_local rows already populated. Nothing to do.")
        return

    print(f"\n  Chunks to process this run: {remaining:,}")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print(f"\n[2/5] Loading {MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    import warnings
    warnings.filterwarnings("ignore")

    t_load = time.time()
    model = SentenceTransformer(MODEL_NAME)
    load_time = time.time() - t_load
    actual_dim = model.get_embedding_dimension()
    print(f"  Loaded in {load_time:.1f}s  |  Dimension: {actual_dim}")

    if actual_dim != LOCAL_DIM:
        print(f"  [ERROR] Model dimension {actual_dim} != expected {LOCAL_DIM}. Aborting.")
        sys.exit(1)

    # ── 3. Fetch IDs to process (strictly NULL rows) ───────────────────────────
    print(f"\n[3/5] Fetching NULL embedding_local chunk IDs...")
    limit_clause = f"LIMIT {args.limit}" if args.limit else ""
    with engine.connect() as conn:
        rows = conn.execute(text(
            f"SELECT id, chunk_text FROM scheme_chunks "
            f"WHERE embedding_local IS NULL "
            f"ORDER BY id ASC {limit_clause}"
        )).fetchall()

    chunk_ids   = [r[0] for r in rows]
    chunk_texts = [r[1] or "" for r in rows]
    total_to_process = len(chunk_ids)
    print(f"  Fetched {total_to_process:,} chunks for processing.")

    # ── 4. Batch setup & Dashboard initialization ─────────────────────────────
    historical_batches = math.floor(c['loc_pop'] / args.batch_size)
    session_batches = math.ceil(total_to_process / args.batch_size)
    total_batches = math.ceil(c['total'] / args.batch_size)

    dashboard = LiveProgressDashboard(
        total_chunks=c['total'],
        initial_completed=c['loc_pop'],
        total_batches=total_batches,
        historical_batches=historical_batches,
        refresh_interval=args.refresh_interval,
        is_dry_run=args.dry_run,
    )

    print(f"\n[4/5] Starting batch processing ({session_batches} batches of {args.batch_size})...\n")
    dashboard.start()

    processed_in_session = 0
    errors = []

    try:
        for batch_idx in range(session_batches):
            if _interrupted:
                break

            bs = batch_idx * args.batch_size
            be = min(bs + args.batch_size, total_to_process)
            batch_ids   = chunk_ids[bs:be]
            batch_texts = chunk_texts[bs:be]

            # Encode with passage prefix
            try:
                embs = model.encode(
                    [f"passage: {t}" for t in batch_texts],
                    batch_size=args.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as e:
                err_msg = f"Batch {batch_idx+1}: encode error: {e}"
                errors.append(err_msg)
                dashboard.add_error(1)
                continue

            # Validate
            if embs.shape != (len(batch_ids), LOCAL_DIM):
                err_msg = f"Batch {batch_idx+1}: unexpected shape {embs.shape}"
                errors.append(err_msg)
                dashboard.add_error(1)
                continue
            if not np.all(np.isfinite(embs)):
                err_msg = f"Batch {batch_idx+1}: non-finite values detected"
                errors.append(err_msg)
                dashboard.add_error(1)
                continue

            # Write to DB (skip if dry-run)
            write_success = True
            if not args.dry_run:
                try:
                    with engine.begin() as conn:
                        for row_id, vec in zip(batch_ids, embs):
                            vec_str = vec_to_pg(vec)
                            conn.execute(text(
                                f"UPDATE scheme_chunks "
                                f"SET embedding_local = '{vec_str}'::vector "
                                f"WHERE id = {int(row_id)}"
                            ))
                except Exception as e:
                    err_msg = f"Batch {batch_idx+1}: DB write error: {e}"
                    errors.append(err_msg)
                    dashboard.add_error(1)
                    write_success = False
                    continue

            if write_success:
                processed_in_session += len(batch_ids)
                # Immediately update progress upon successful DB commit
                dashboard.update_batch(
                    chunks_committed=len(batch_ids),
                    batch_index=batch_idx + 1
                )

    finally:
        dashboard.stop()
        final_stats = dashboard.get_stats()

    # ── Handle Interrupt vs Completion ────────────────────────────────────────
    if _interrupted:
        print("\n" + "=" * 26)
        print("GENERATION INTERRUPTED")
        print("=" * 26)
        print(f"Total chunks:         {final_stats['total']:,}")
        print(f"Committed chunks:     {final_stats['completed']:,} (preserved in DB)")
        print(f"Remaining chunks:     {final_stats['remaining']:,}")
        print(f"Elapsed time:         {fmt_time(final_stats['elapsed'])}")
        print(f"Errors:               {final_stats['errors']}")
        print("=" * 26)
        print("\n[INFO] All committed embeddings in PostgreSQL remain safely saved.")
        print("[INFO] The script can be resumed at any time by running:")
        print("       python scripts/18_generate_local_embeddings.py")
        sys.exit(0)

    # Permanent final summary lines upon completion
    print("\n" + "=" * 26)
    print("LOCAL EMBEDDING COMPLETE")
    print("=" * 26)
    print(f"Total:         {final_stats['total']:,}")
    print(f"Completed:     {final_stats['completed']:,}")
    print(f"Remaining:     {final_stats['remaining']:,}")
    print(f"Duration:      {fmt_time(final_stats['elapsed'])}")
    avg_speed_str = f"{final_stats['speed']:.1f} chunks/sec" if final_stats["speed"] > 0 else "N/A"
    print(f"Average speed: {avg_speed_str}")
    print(f"Errors:        {final_stats['errors']}")
    print("=" * 26)

    # ── 5. Post-generation integrity check ────────────────────────────────────
    print(f"\n[5/5] Post-generation integrity check...")
    with engine.connect() as conn:
        c_final = get_counts(conn)
        dups = conn.execute(text(
            "SELECT count(*) FROM ("
            "  SELECT chunk_id, count(*) FROM scheme_chunks "
            "  GROUP BY chunk_id HAVING count(*) > 1"
            ") t"
        )).scalar()

    print(f"  Total chunks:            {c_final['total']:,}")
    print(f"  Gemini populated:        {c_final['gem_pop']:,}")
    print(f"  Gemini NULL:             {c_final['gem_null']:,}")
    print(f"  embedding_local pop:     {c_final['loc_pop']:,}")
    print(f"  embedding_local NULL:    {c_final['loc_null']:,}")
    print(f"  Duplicate chunk_id rows: {dups}")

    # ── 6. Smoke tests ────────────────────────────────────────────────────────
    smoke_results = []
    if not args.dry_run and c_final['loc_pop'] > 0:
        print(f"\n  Retrieval smoke tests...")
        smoke_queries = [
            ("What scholarships are available for students?", "EN"),
            ("महिलाओं के लिए सरकारी योजना कौन सी है?", "HI"),
            ("ખેડૂતો માટે કઈ સહાય યોજના ઉપલબ્ધ છે?", "GU"),
        ]
        with engine.connect() as conn:
            for q_text, lang in smoke_queries:
                q_vec = model.encode(f"query: {q_text}", normalize_embeddings=True)
                q_str = vec_to_pg(q_vec)
                sql = (
                    f"SELECT sc.id, sc.metadata->>'scheme_name' as sname, "
                    f"1 - (sc.embedding_local <=> '{q_str}'::vector) as sim "
                    f"FROM scheme_chunks sc "
                    f"WHERE sc.embedding_local IS NOT NULL "
                    f"ORDER BY sc.embedding_local <=> '{q_str}'::vector ASC "
                    f"LIMIT 3"
                )
                rows = conn.execute(text(sql)).fetchall()
                print(f"\n  [{lang}] '{q_text[:55]}'")
                for rank, r in enumerate(rows, 1):
                    print(f"    {rank}. sim={r[2]:.4f}  {(r[1] or 'N/A')[:60]}")
                smoke_results.append({
                    "lang": lang,
                    "query": q_text,
                    "top3": [{"id": r[0], "scheme": r[1], "sim": round(float(r[2]), 4)} for r in rows]
                })

    # Save summary JSON
    import json
    summary = {
        "mode": "dry_run" if args.dry_run else "live",
        "model": MODEL_NAME,
        "dimension": LOCAL_DIM,
        "batch_size": args.batch_size,
        "chunks_processed": processed_in_session,
        "elapsed_seconds": round(final_stats["elapsed"], 1),
        "avg_rate": round(final_stats["speed"], 1),
        "errors": errors,
        "interrupted": _interrupted,
        "counts_before": c,
        "counts_after": c_final,
        "duplicate_rows": dups,
        "smoke_tests": smoke_results,
        "overall": "PASS" if (len(errors) == 0 and dups == 0 and c_final['gem_pop'] == 1260) else "FAIL",
    }
    out = PROJECT_ROOT / "reports" / f"local_emb_gen_{'dryrun' if args.dry_run else 'full'}_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

if __name__ == "__main__":
    main()
