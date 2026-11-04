"""Text-free evaluation records, with opaque identities and exact numerical outcomes."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from .metrics import ap_at_k_single_heldout, recall_at_k_single_heldout

CONDITIONS = ("C0", "C1", "C2", "C4")
STEERS = {"authority", "scarcity", "social_proof", "exclusivity", "identity_signaling",
          "storytelling", "comparative_framing"}
CONTEXT_FIELDS = {"dataset", "instance_id", "target_cluster_id", "target_initial_rank",
                  "target_stratum", "gold_stage1_rank", "cohort_origin"}
TARGET_FIELDS = {"target_cluster_id", "steers", "description_lengths"}
OBSERVATION_FIELDS = {
    "condition", "parsed_scores", "valid", "fallback", "abstained", "ap10", "recall10",
    "model_id", "provider_name", "provider_backend", "finish_reason", "input_tokens",
    "output_tokens", "reasoning_tokens", "cached_input_tokens", "latency_ms", "model",
    "instance_id", "dataset", "replicate", "profile_id", "ranking_indices", "provider_error",
    "api_embedded_error", "refusal_recorded", "refusal_text_heuristic", "short_incomplete_output",
    "route", "exact_stage1_score_copy", "stage1_score_match_fraction",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


class Benchmark:
    def __init__(self, directory):
        self.root = Path(directory)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["schema_version"] != "seller_audit_results_v1":
            raise ValueError("Expected a results-only release")
        self.verify_files()
        self.contexts, self.variants = {}, {}
        self.profiles = json.loads((self.root / "profiles.json").read_text(encoding="utf-8"))
        for dataset in self.manifest["datasets"]:
            for row in read_jsonl(self.root / dataset / "contexts.jsonl.gz"):
                key = (dataset, row["instance_id"])
                if set(row) != CONTEXT_FIELDS or key in self.contexts or row["dataset"] != dataset:
                    raise ValueError("Invalid or duplicate numeric context")
                self.contexts[key] = row
            for row in read_jsonl(self.root / dataset / "target_metadata.jsonl.gz"):
                key = (dataset, row["target_cluster_id"])
                if set(row) != TARGET_FIELDS or key in self.variants:
                    raise ValueError("Invalid or duplicate target metadata")
                self.variants[key] = row

    def observations(self):
        for model in self.manifest["models"]:
            for row in read_jsonl(self.root / "observations" / (model + ".jsonl.gz")):
                if set(row) != OBSERVATION_FIELDS or row["model"] != model:
                    raise ValueError("Observation schema differs from numeric allowlist")
                yield row

    def verify_files(self):
        required = {"profiles.json", "LICENSE_DATA.md"}
        for d in self.manifest["datasets"]:
            required.update((f"{d}/contexts.jsonl.gz", f"{d}/target_metadata.jsonl.gz"))
        required.update(f"observations/{m}.jsonl.gz" for m in self.manifest["models"])
        if set(self.manifest["files"]) != required:
            raise ValueError("Unexpected/missing data member")
        if {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()} != required | {"manifest.json"}:
            raise ValueError("Unexpected file in results-only data directory")
        for relative, expected in self.manifest["files"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root.resolve()) or file_hash(path) != expected:
                raise ValueError(f"Missing or altered results file: {relative}")

    def hydrate(self, observation):
        """Use local candidate indices for ranking math; retain target clustering."""
        o = observation.copy()
        c = self.contexts[o["dataset"], o["instance_id"]]
        target = c["target_cluster_id"]
        ids = [f"candidate_{i:02d}" for i in range(20)]
        ids[c["target_initial_rank"] - 1] = target
        gold_rank = c["gold_stage1_rank"]
        gold = ids[gold_rank - 1] if gold_rank is not None and gold_rank <= 20 else "outside_candidate_head"
        settings = self.profiles[o["profile_id"]]["settings"]
        # These are comparison-group identities, not reconstructed prompt hashes.
        input_group = digest([o["dataset"], o["instance_id"], o["condition"]])
        o.update(candidate_item_ids=ids, final_ranking=[ids[i] for i in o["ranking_indices"]],
                 heldout_item_id=gold, target_item_id=target, target_initial_rank=c["target_initial_rank"],
                 target_stratum=c["target_stratum"], gold_stage1_rank=gold_rank,
                 cohort_origin=c["cohort_origin"], victim_settings=settings, victim_settings_digest=digest(settings),
                 benchmark_digest=self.manifest["scientific_digest"], execution_id=o["profile_id"],
                 run_fingerprint=o["profile_id"], victim_template_hash="fixed_historical_protocol",
                 prompt_hash=input_group,
                 request_key=digest([o["model"], o["instance_id"], o["dataset"], o["condition"], o["replicate"]]))
        o["ap10"] = ap_at_k_single_heldout(gold, o["final_ranking"])
        o["recall10"] = recall_at_k_single_heldout(gold, o["final_ranking"])
        return o
