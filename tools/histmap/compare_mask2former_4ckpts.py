from pathlib import Path
import re


RUNS = [
    ("r50", Path("work_dirs/histmap_mask2former_ckpt_r50_20k")),
    ("r101", Path("work_dirs/histmap_mask2former_ckpt_r101_20k")),
    ("swin_t", Path("work_dirs/histmap_mask2former_ckpt_swin_t_20k")),
    ("swin_s", Path("work_dirs/histmap_mask2former_ckpt_swin_s_20k")),
]

BEST_RE = re.compile(
    r"The best checkpoint with ([0-9.]+) coco/segm_mAP at ([0-9]+) iter "
    r"is saved to (\S+)")
VAL_RE = re.compile(r"Iter\(val\).*?coco/segm_mAP: ([0-9.]+)")


def rel(path):
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def latest_log(work_dir):
    logs = sorted(work_dir.glob("*/*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def summarize(name, work_dir):
    log_path = latest_log(work_dir)
    if log_path is None:
        return [name, "pending", "-", rel(work_dir), "-"]

    best_score = None
    best_iter = "-"
    best_ckpt = "-"
    fallback_score = None

    for line in log_path.read_text(errors="replace").splitlines():
        best_match = BEST_RE.search(line)
        if best_match:
            score = float(best_match.group(1))
            if best_score is None or score >= best_score:
                best_score = score
                best_iter = best_match.group(2)
                best_ckpt = rel(work_dir / best_match.group(3).rstrip("."))
            continue

        val_match = VAL_RE.search(line)
        if val_match:
            score = float(val_match.group(1))
            if fallback_score is None or score > fallback_score:
                fallback_score = score

    if best_score is None and fallback_score is not None:
        best_score = fallback_score

    score_text = "-" if best_score is None else f"{best_score:.4f}"
    return [name, score_text, best_iter, rel(log_path), best_ckpt]


def main():
    rows = [summarize(name, work_dir) for name, work_dir in RUNS]
    headers = ["checkpoint", "best_coco/segm_mAP", "iter", "log", "best_checkpoint"]
    widths = [
        max(len(str(row[i])) for row in [headers] + rows)
        for i in range(len(headers))
    ]

    def fmt(row):
        return "  ".join(str(value).ljust(widths[i])
                         for i, value in enumerate(row))

    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))
    for row in rows:
        print(fmt(row))


if __name__ == "__main__":
    main()
