"""Pack each simulation run into a per-run .tar.gz and push them to a Hugging Face dataset.

Each run folder under the data dir holds ~1000 small coord dumps, so we archive
one tarball per run (1000 files instead of ~800k) for a fast, reliable upload.

Auth uses the cached `hf` token (run `hf auth login` once if needed). Uploads go
to the dataset repo via the `hf` CLI, which is bundled separately from this
project's Python, so we shell out to it rather than importing huggingface_hub.

    python push_to_huggingface.py                 # tar (parallel) then upload
    python push_to_huggingface.py --no-upload      # only build the archives
    python push_to_huggingface.py --no-archive     # only upload existing archives
    python push_to_huggingface.py --workers 8      # cap parallel tar jobs
"""
import argparse
import concurrent.futures
import os
import subprocess
import tarfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "scratch_local")
DEFAULT_ARCHIVE_DIR = os.path.join(REPO_ROOT, "data", "hf_archives")
HF_REPO = "ljding94/MLFF_CGMD"


def make_archive(run_dir, archive_dir):
    """Tar+gzip a single run folder into archive_dir. Skips if already built.

    Writes to a temp file first and renames on success so an interrupted run
    never leaves a half-written, resumable-looking archive behind.
    """
    name = os.path.basename(os.path.normpath(run_dir))
    out_path = os.path.join(archive_dir, name + ".tar.gz")
    if os.path.exists(out_path):
        return name, False  # already archived
    tmp_path = out_path + ".tmp"
    with tarfile.open(tmp_path, "w:gz") as tar:
        tar.add(run_dir, arcname=name)
    os.replace(tmp_path, out_path)
    return name, True


def build_archives(data_dir, archive_dir, workers):
    os.makedirs(archive_dir, exist_ok=True)
    run_dirs = sorted(
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    print(f"Archiving {len(run_dirs)} run folders -> {archive_dir} "
          f"with {workers} parallel workers")

    built = skipped = 0
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(make_archive, rd, archive_dir): rd for rd in run_dirs}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            created = future.result()[1]
            if created:
                built += 1
            else:
                skipped += 1
            if done % 50 == 0 or done == len(run_dirs):
                print(f"  [{done}/{len(run_dirs)}] built={built} skipped={skipped}")
    print(f"Archives ready: {built} built, {skipped} already existed")


def upload(archive_dir, repo):
    print(f"Creating dataset repo {repo} (ok if it already exists)")
    subprocess.run(
        ["hf", "repo", "create", repo, "--repo-type", "dataset"],
        check=False,
    )
    print(f"Uploading {archive_dir} -> {repo}")
    subprocess.run(
        ["hf", "upload-large-folder", repo, archive_dir, "--repo-type", "dataset"],
        check=True,
    )
    print(f"Done. View at https://huggingface.co/datasets/{repo}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--archive_dir", default=DEFAULT_ARCHIVE_DIR)
    p.add_argument("--repo", default=HF_REPO)
    p.add_argument("--workers", type=int, default=None,
                   help="parallel tar jobs (default: os.cpu_count())")
    p.add_argument("--no-archive", action="store_true", help="skip tarring; upload existing archives")
    p.add_argument("--no-upload", action="store_true", help="only build archives, do not upload")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.no_archive:
        build_archives(args.data_dir, args.archive_dir, args.workers or os.cpu_count() or 1)
    if not args.no_upload:
        upload(args.archive_dir, args.repo)
