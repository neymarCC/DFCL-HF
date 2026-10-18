"""Patient-level bootstrap 95% CI for key CL table metrics. Shared resample indices."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

B = 2000
SEED = 42
METHODS = ("naive", "ewc", "lwf", "api", "dkd", "lwht", "dualcp", "dfcl_hf")

METHOD_ROOTS = {
    "mimic": {
        "naive": ROOT / "runs" / "baselines" / "naive" / "mimic",
        "ewc": ROOT / "runs" / "baselines" / "ewc" / "mimic",
        "lwf": ROOT / "runs" / "baselines" / "lwf" / "mimic",
        "api": ROOT / "runs" / "baselines" / "api" / "mimic",
        "dkd": ROOT / "runs" / "baselines" / "dkd" / "mimic",
        "lwht": ROOT / "runs" / "baselines" / "lwht" / "mimic",
        "dualcp": ROOT / "runs" / "baselines" / "dualcp" / "mimic",
        "dfcl_hf": ROOT / "runs" / "dfcl_hf",
    },
    "echonext": {
        "naive": ROOT / "runs" / "baselines" / "naive" / "echonext",
        "ewc": ROOT / "runs" / "baselines" / "ewc" / "echonext",
        "lwf": ROOT / "runs" / "baselines" / "lwf" / "echonext",
        "api": ROOT / "runs" / "baselines" / "api" / "echonext",
        "dkd": ROOT / "runs" / "baselines" / "dkd" / "echonext",
        "lwht": ROOT / "runs" / "baselines" / "lwht" / "echonext",
        "dualcp": ROOT / "runs" / "baselines" / "dualcp" / "echonext",
        "dfcl_hf": ROOT / "runs" / "dfcl_hf_echonext",
    },
}


def pred_path(root: Path, model_stage: int, test_stage: int) -> Path:
    return root / "predictions" / f"model_stage_{model_stage:02d}_test_stage_{test_stage:02d}.csv"


def future_pred_path(root: Path) -> Path:
    for name in (
        "predictions.csv",
        "s6_future_2022_predictions.csv",
        "predictions_s6.csv",
    ):
        path = root / "future_2022" / name
        if path.exists():
            return path
    return root / "future_2022" / "predictions.csv"


def load_pred(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def pack_frame(frame: pd.DataFrame) -> dict:
    y = frame["true_label"].to_numpy(dtype=np.int64)
    p = frame["predicted_probability"].to_numpy(dtype=np.float64)
    pid = frame["patient_id"].to_numpy()
    patients, inverse = np.unique(pid, return_inverse=True)
    order = np.argsort(inverse, kind="mergesort")
    inv_sorted = inverse[order]
    cuts = np.flatnonzero(inv_sorted[1:] != inv_sorted[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(inv_sorted)]))
    groups = [order[s:e] for s, e in zip(starts, ends)]
    one_row = all(len(g) == 1 for g in groups)
    row_of = np.array([g[0] for g in groups], dtype=np.int64) if one_row else None
    return {
        "y": y,
        "p": p,
        "patients": patients,
        "groups": groups,
        "one_row": one_row,
        "row_of": row_of,
    }


def take_rows(packed: dict, idx: np.ndarray) -> np.ndarray:
    if packed["one_row"]:
        return packed["row_of"][idx]
    return np.concatenate([packed["groups"][int(i)] for i in idx])


def metric_pack(packed: dict, idx: np.ndarray, kind: str) -> float:
    rows = take_rows(packed, idx)
    y = packed["y"][rows]
    p = packed["p"][rows]
    return auroc(y, p) if kind == "auroc" else auprc(y, p)


def aligned_idx(canonical: np.ndarray, local_n: int, idx: np.ndarray) -> np.ndarray:
    if idx.shape[0] == local_n:
        return idx
    return idx % local_n


def ensure_indices(dataset: str, n_stages: int, canonical_root: Path) -> dict[str, dict]:
    index_dir = ROOT / "runs" / "baselines" / "_bootstrap_indices" / dataset
    index_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    store: dict[str, dict] = {}
    keys = list(range(n_stages)) + (["future"] if dataset == "echonext" else [])
    for test_stage in keys:
        if test_stage == "future":
            path = future_pred_path(canonical_root)
            key = "future"
        else:
            path = pred_path(canonical_root, 0, int(test_stage))
            key = str(test_stage)
        if not path.exists():
            print(f"[warn] missing canonical preds {path}", flush=True)
            continue
        print(f"[index] {dataset} test={key}", flush=True)
        packed = pack_frame(load_pred(path))
        npy = index_dir / f"test_{key}_patient_idx.npy"
        n = len(packed["patients"])
        if npy.exists():
            idx = np.load(npy, allow_pickle=False)
        else:
            idx = rng.integers(0, n, size=(B, n), dtype=np.int64)
            np.save(npy, idx)
        store[key] = {"n": n, "idx": idx}
    return store


def ci_from_samples(point: float, samples: np.ndarray) -> dict[str, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "point": float(point),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
    }


def fmt(triple: dict[str, float]) -> str:
    return f"{triple['point']:.4f} ({triple['ci_low']:.4f}, {triple['ci_high']:.4f})"


def method_ready(root: Path, n_stages: int) -> bool:
    final = n_stages - 1
    return pred_path(root, final, 0).exists() and (root / "matrices" / "auroc_continual_metrics.csv").exists()


def bootstrap_method(dataset: str, method: str, n_stages: int, index_meta: dict) -> dict | None:
    root = METHOD_ROOTS[dataset][method]
    if not method_ready(root, n_stages):
        print(f"[skip] {method}/{dataset} incomplete: {root}", flush=True)
        return None
    print(f"[boot] {method}/{dataset}", flush=True)
    T = n_stages - 1
    packed: dict[tuple[int, int], dict] = {}
    for test_j in range(n_stages):
        for model_i in range(test_j, n_stages):
            path = pred_path(root, model_i, test_j)
            if path.exists():
                packed[(model_i, test_j)] = pack_frame(load_pred(path))

    def boot(model_i: int, test_j: int, b: int) -> float:
        key = str(test_j)
        pk = packed[(model_i, test_j)]
        idx = aligned_idx(None, len(pk["patients"]), index_meta[key]["idx"][b])
        return metric_pack(pk, idx, "auroc")

    js = [j for j in range(n_stages) if (T, j) in packed]
    point_ap = float(np.mean([auroc(packed[(T, j)]["y"], packed[(T, j)]["p"]) for j in js]))
    samples_ap = np.empty(B)
    for b in range(B):
        samples_ap[b] = float(np.mean([boot(T, j, b) for j in js]))

    bwt_js = [j for j in range(T) if (T, j) in packed and (j, j) in packed]
    point_bwt = float(
        np.mean(
            [
                auroc(packed[(T, j)]["y"], packed[(T, j)]["p"]) - auroc(packed[(j, j)]["y"], packed[(j, j)]["p"])
                for j in bwt_js
            ]
        )
    )
    samples_bwt = np.empty(B)
    for b in range(B):
        samples_bwt[b] = float(np.mean([boot(T, j, b) - boot(j, j, b) for j in bwt_js]))

    forget_js = [j for j in range(T) if (T, j) in packed]

    def forget_point() -> float:
        vals = []
        for j in forget_js:
            prev = [auroc(packed[(k, j)]["y"], packed[(k, j)]["p"]) for k in range(j, T) if (k, j) in packed]
            if prev:
                vals.append(max(prev) - auroc(packed[(T, j)]["y"], packed[(T, j)]["p"]))
        return float(np.mean(vals)) if vals else float("nan")

    samples_fg = np.empty(B)
    for b in range(B):
        vals = []
        for j in forget_js:
            prev = [boot(k, j, b) for k in range(j, T) if (k, j) in packed]
            if prev:
                vals.append(max(prev) - boot(T, j, b))
        samples_fg[b] = float(np.mean(vals)) if vals else float("nan")

    metrics = pd.read_csv(root / "matrices" / "auroc_continual_metrics.csv")
    last = metrics[metrics["model_stage"] == T].iloc[0]
    samples_final = np.array([boot(T, T, b) for b in range(B)])
    result = {
        "method": method,
        "dataset": dataset,
        "final_avg_auroc": ci_from_samples(point_ap, samples_ap),
        "bwt": ci_from_samples(point_bwt, samples_bwt),
        "forgetting": ci_from_samples(forget_point(), samples_fg),
        "fwt_point": float(last["forward_transfer"]),
        "final_stage_auroc": ci_from_samples(auroc(packed[(T, T)]["y"], packed[(T, T)]["p"]), samples_final),
    }
    if dataset == "echonext":
        fut = future_pred_path(root)
        if fut.exists() and "future" in index_meta:
            ff = pack_frame(load_pred(fut))
            idx_all = index_meta["future"]["idx"]
            samples_fa = np.empty(B)
            samples_fp = np.empty(B)
            for b in range(B):
                local = aligned_idx(None, len(ff["patients"]), idx_all[b])
                samples_fa[b] = metric_pack(ff, local, "auroc")
                samples_fp[b] = metric_pack(ff, local, "auprc")
            result["future_auroc"] = ci_from_samples(auroc(ff["y"], ff["p"]), samples_fa)
            result["future_auprc"] = ci_from_samples(auprc(ff["y"], ff["p"]), samples_fp)
    print(f"[done-boot] {method}/{dataset}", flush=True)
    return result


def main() -> None:
    out_dir = ROOT / "runs" / "baselines" / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[start] building shared bootstrap indices", flush=True)
    mimic_idx = ensure_indices("mimic", 11, METHOD_ROOTS["mimic"]["dfcl_hf"])
    echo_idx = ensure_indices("echonext", 7, METHOD_ROOTS["echonext"]["dfcl_hf"])
    mimic_rows = []
    echo_rows = []
    payload = {"mimic": {}, "echonext": {}}
    for method in METHODS:
        m = bootstrap_method("mimic", method, 11, mimic_idx)
        if m:
            payload["mimic"][method] = m
            mimic_rows.append(
                {
                    "Method": method,
                    "Final Avg AUROC (95% CI)": fmt(m["final_avg_auroc"]),
                    "BWT (95% CI)": fmt(m["bwt"]),
                    "Avg Forgetting (95% CI)": fmt(m["forgetting"]),
                    "Final-stage AUROC (95% CI)": fmt(m["final_stage_auroc"]),
                }
            )
            pd.DataFrame(mimic_rows).to_csv(out_dir / "table_a_mimic.csv", index=False)
        e = bootstrap_method("echonext", method, 7, echo_idx)
        if e:
            payload["echonext"][method] = e
            echo_rows.append(
                {
                    "Method": method,
                    "Final Avg AUROC (95% CI)": fmt(e["final_avg_auroc"]),
                    "BWT (95% CI)": fmt(e["bwt"]),
                    "Avg Forgetting (95% CI)": fmt(e["forgetting"]),
                    "FWT": f"{e['fwt_point']:.4f}",
                    "2022 AUROC (95% CI)": fmt(e["future_auroc"]) if "future_auroc" in e else "NA",
                    "2022 AUPRC (95% CI)": fmt(e["future_auprc"]) if "future_auprc" in e else "NA",
                }
            )
            pd.DataFrame(echo_rows).to_csv(out_dir / "table_b_echonext.csv", index=False)
        with (out_dir / "bootstrap_payload.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
    print("\nTable A — MIMIC continual learning comparison", flush=True)
    print(pd.DataFrame(mimic_rows).to_string(index=False), flush=True)
    print("\nTable B — EchoNext temporal continual learning comparison", flush=True)
    print(pd.DataFrame(echo_rows).to_string(index=False), flush=True)
    print(f"[saved] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
