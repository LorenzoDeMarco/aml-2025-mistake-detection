import torch
from pathlib import Path
import json

pt_dir = Path("substep3_3/output_3_3")
recordings = json.load(open("./captaincook_actionformer_annotations/combined/recordings.json"))["database"]

# Controlla similarità medie per video correct vs error
correct_sims, error_sims = [], []
for pt in pt_dir.glob("*.pt"):
    d = torch.load(pt, map_location="cpu")
    vid = pt.stem
    if vid not in recordings:
        continue
    sims = d.get("match_similarities", [])
    if not sims:
        continue
    avg_sim = sum(sims) / len(sims)
    label = recordings[vid]
    anns = label.get("annotations", [])
    has_error = any(a.get("has_error", False) for a in anns)
    if has_error:
        error_sims.append(avg_sim)
    else:
        correct_sims.append(avg_sim)

print(f"Avg sim CORRECT: {sum(correct_sims)/len(correct_sims):.3f} ({len(correct_sims)} video)")
print(f"Avg sim ERROR:   {sum(error_sims)/len(error_sims):.3f} ({len(error_sims)} video)")