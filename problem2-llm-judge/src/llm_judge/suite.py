"""Load test suites (YAML or JSON) and adversarial probes into TestCase objects."""

import json
from pathlib import Path

import yaml

from .config import PROJECT_ROOT
from .rubric import criteria_for
from .schema import TestCase

DEFAULT_SUITE = PROJECT_ROOT / "suites" / "core_suite.yaml"
DEFAULT_PROBES = PROJECT_ROOT / "probes" / "adversarial.yaml"


def _read(path: Path) -> dict | list:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)


def load_suite(path: Path = DEFAULT_SUITE) -> list[TestCase]:
    data = _read(Path(path))
    raw = data["cases"] if isinstance(data, dict) else data
    cases = [TestCase(**c) for c in raw]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in {path}")
    for c in cases:
        criteria_for(c.criteria)  # validates criterion names
    return cases


def load_probes(path: Path = DEFAULT_PROBES, suite: list[TestCase] | None = None) -> tuple[list[dict], list[dict]]:
    """Return (probes, pairs). Each probe dict has 'case' (a TestCase with the probe output),
    'family', 'expect' and 'base' (the suite case it derives from, if any)."""
    data = _read(Path(path))
    by_id = {c.id: c for c in (suite or load_suite())}
    probes = []
    for p in data["probes"]:
        if "base" in p:
            base = by_id[p["base"]]
            case = base.model_copy(update={"id": p["id"], "model_output": p["model_output"], "gold_label": None})
        else:
            case = TestCase(id=p["id"], category="probe", model_output=p["model_output"], **p["case"])
        probes.append({"id": p["id"], "family": p["family"], "expect": p["expect"], "base": p.get("base"), "case": case})
    probe_by_id = {p["id"]: p for p in probes}

    def output(ref: str, base: str) -> str:
        return by_id[base].model_output if ref == "suite" else probe_by_id[ref]["case"].model_output

    pairs = [{**pr, "case": by_id[pr["base"]], "better_output": output(pr["better"], pr["base"]),
              "worse_output": output(pr["worse"], pr["base"])} for pr in data.get("pairs", [])]
    return probes, pairs
