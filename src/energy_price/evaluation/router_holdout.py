"""Fail-closed, one-shot holdout protocol for the final configurable pool.

Readiness reads only missingness and provenance, never holdout performance.
Finality is an explicit source-backed attestation, not inferred from a date or
the mere existence of a price file. A failed started run remains consumed.
"""

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from energy_price.features.dataset import TARGET, build_table, load_inputs, quarter_hours
from energy_price.models.shared.forecast_pool import generate_pool, wide_pool
from energy_price.models.time_router import model as config
from energy_price.paths import ROOT


def readiness(table):
    expected = quarter_hours(list(pd.date_range("2026-09-01", "2026-10-31")))
    actual = expected[["timestamp_utc"]].merge(table, on="timestamp_utc", how="left", validate="one_to_one")
    features = ["radiation", "radiation_change", "holiday_pv_share"]
    required = [TARGET, *features]
    missing = {c: int(actual[c].isna().sum()) for c in required}
    digest = sha256(pd.util.hash_pandas_object(actual[["timestamp_utc", *required]], index=False).values.tobytes()).hexdigest()
    return {"n_expected_quarters": len(expected), "missing": missing,
            "complete": not any(missing.values()), "data_fingerprint": digest,
            "missing_direction": int(actual["short"].isna().sum()) if "short" in actual else len(actual)}


def check_final_evidence(path, ready, today):
    if path is None:
        raise ValueError("source-backed --final-data-evidence JSON is required")
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    if (evidence.get("final_prices_confirmed") is not True
            or set(evidence.get("months", [])) != {"2026-09", "2026-10"}
            or not evidence.get("source_url") or not evidence.get("verified_by")
            or not evidence.get("explanation")
            or evidence.get("data_fingerprint") != ready["data_fingerprint"]):
        raise ValueError("final-data evidence is incomplete or refers to different data")
    verified = pd.Timestamp(evidence.get("verified_on"))
    if pd.isna(verified) or verified.normalize() > today.normalize() or verified < pd.Timestamp("2026-11-01"):
        raise ValueError("invalid final-data verification date")
    if not ready["complete"] or ready["missing_direction"]:
        raise ValueError("holdout data/direction labels are incomplete")
    return evidence


def claim_once(path, record):
    """Atomic creation: a second invocation cannot overwrite a started run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
    except FileExistsError as exc:
        raise RuntimeError("holdout already started; no second run or retuning permitted") from exc


def run(args):
    from energy_price.evaluation.time_router import (DOC, DESIGN, NOMINATION, dump, pool_phase,
        read_design, require_committed, routed_history, summarise, json_fingerprint)
    from energy_price.evaluation.backtest import run_info

    today = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).normalize()
    table = build_table(load_inputs(), days=list(pd.date_range("2026-01-01", "2026-10-31")))
    ready = readiness(table)
    dump(args.out / "holdout_readiness.json", ready)
    if args.step == "readiness":
        print(json.dumps(ready, indent=2))
        return
    if today < pd.Timestamp(config.POOL_FREEZE_DATE):
        raise ValueError(f"final pool cannot freeze/run before {config.POOL_FREEZE_DATE}")
    design = read_design()
    require_committed(NOMINATION)
    nominated = json.loads(NOMINATION.read_text(encoding="utf-8"))
    if nominated["design_sha256"] != json_fingerprint(DESIGN):
        raise ValueError("nomination belongs to an older pool/design; repeat allowed development phases")
    evidence = check_final_evidence(args.final_data_evidence, ready, today)
    frozen_path = DOC / "pool_freeze.json"
    frozen = {"design_sha256": json_fingerprint(DESIGN),
              "nomination_sha256": json_fingerprint(NOMINATION),
              "candidate": nominated["candidate"], "pool": design["pool"],
              "data_fingerprint": ready["data_fingerprint"], "final_data_evidence": evidence,
              "freeze_date": str(today.date()), "info": run_info()}
    if args.step == "freeze":
        claim_once(frozen_path, frozen)
        print("Pool frozen. Commit pool_freeze.json before the single holdout invocation.")
        return
    require_committed(frozen_path)
    previous = json.loads(frozen_path.read_text(encoding="utf-8"))
    for key in ("design_sha256", "nomination_sha256", "data_fingerprint"):
        if previous[key] != frozen[key]:
            raise ValueError(f"changed {key} after final pool freeze")
    if subprocess_dirty():
        raise ValueError("commit the implementation before the one-shot holdout")
    ledger_path = DOC / "holdout_protocol.json"
    record = {"status": "started", "freeze_sha256": sha256(frozen_path.read_bytes()).hexdigest(), "info": run_info()}
    claim_once(ledger_path, record)
    # From here the ledger is consumed, even if fitting fails. It cannot be
    # silently reset by a CLI flag. Investigation must leave a visible record.
    frames = [pool_phase(table, p, args.out, args.jobs) for p in ("warm", "tune", "test")]
    holdout = generate_pool(table, list(pd.date_range("2026-09-01", "2026-10-31")), design["pool"], args.jobs)
    holdout.to_parquet(args.out / "holdout_pool.parquet", index=False)
    wide = wide_pool(pd.concat([*frames, holdout], ignore_index=True), list(design["pool"]))
    final_design = {**design, "candidates": [nominated["candidate"]]}
    forecasts, decisions = routed_history(wide, final_design, "2026-10-31")
    decisions.to_csv(args.out / "holdout_decisions.csv", index=False)
    scores, pairs = summarise(forecasts, final_design, "holdout", args.out, args.track)
    experts = []
    for name, group in holdout.groupby("model"):
        # Exact same admissible quarters as the architecture comparison.
        keys = forecasts[forecasts.delivery_day >= "2026-09-01"].timestamp_utc.unique()
        experts.append({"model": name, **score_expert(group[group.timestamp_utc.isin(keys)])})
    pd.DataFrame(experts).to_csv(args.out / "holdout_expert_scores.csv", index=False)
    pair = pairs[pairs.model == "calibrated_" + nominated["candidate"]].iloc[0]
    candidate_name = "calibrated_" + nominated["candidate"]
    # The score is quarter-hour weighted, whereas inference averages days.
    # These are distinct on October's 100-quarter-hour DST day: require both.
    lower_pinball = scores.loc[candidate_name, "pinball"] < scores.loc["calibrated_default", "pinball"]
    accepted = bool(lower_pinball and pair.ci95_high < 0 and pair.coverage_difference >= 0)
    record.update(status="completed", decision="time_architecture" if accepted else "calibrated_single",
                  comparison=pair.to_dict(), candidate=nominated["candidate"])
    dump(ledger_path, record)
    with (DOC / "README.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\nHoldout {today.date()}: `{record['decision']}`; Protokoll: `holdout_protocol.json`.\n")
    print(json.dumps(record, indent=2, default=str))


def subprocess_dirty():
    import subprocess
    result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, check=True)
    return bool(result.stdout.strip())


def score_expert(frame):
    from energy_price.evaluation.scores import score
    return score(frame)
