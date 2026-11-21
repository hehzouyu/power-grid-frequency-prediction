"""Create stage settings without simulating or inspecting confirmation data."""
import argparse
import json
from pathlib import Path
base=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("stage",choices=["dev","confirm"])
args=p.parse_args()
c=json.loads((base/"nadir_quality_protocol.json").read_text())
q=json.loads((base/"nadir_pair_protocol.json").read_text())
s=args.stage
out=base/"out"/"nadir-quality"/s
assert not out.exists(), f"Must preserve existing run: {out}"
q.update(version="quality-"+s,reference_seed=c[s+"_reference_seed"],noise_seed=c[s+"_noise_seed"],
         references_per_r_band=c[s+"_pair_refs_per_band"],noise_repeats=1)
out.mkdir(parents=True)
(out/"protocol.json").write_text(json.dumps(q,indent=2))
(out/"stage.json").write_text(json.dumps(dict(single_per_band=c[s+"_single_per_band"],
                                              noise_seed=c[s+"_noise_seed"],repeats=c[s+"_noise_repeats"]),indent=2))
