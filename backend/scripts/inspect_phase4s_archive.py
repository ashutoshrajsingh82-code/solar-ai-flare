import json
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from scripts.index_real_observations import RealDatasetIndexer

p = Path(r"C:\Users\ASHUTOSH\Downloads\solexs_2026Oct25T200126271.zip")
indexer = RealDatasetIndexer([p])
res = indexer.index_all()

print(f"Total products indexed: {res['total_products_indexed']}")
print(f"Eligible for forecasting: {res['eligible_for_forecasting']}")
print(f"Excluded products: {res['excluded_products']}")
print(f"Duplicate groups: {len(res['duplicate_groups'])}")
print(f"Overlapping conflicts: {res['overlapping_interval_conflicts']}")

print("\nEligible SDD2 Lightcurves:")
total_finite = 0
total_rec = 0
total_segs = 0
for item in res["inventory"]:
    if item["eligibility_decision"] == "ELIGIBLE":
        total_finite += item["finite_records"]
        total_rec += item["total_records"]
        total_segs += item["num_contiguous_segments"]
        print(f"  - {item['observation_date']}: records={item['finite_records']}/{item['total_records']}, "
              f"duty={item['duty_cycle_pct']}%, segs={item['num_contiguous_segments']}, "
              f"mean={item['mean_count_rate']} c/s, max={item['max_count_rate']} c/s, "
              f"path={item['member_path']}")

print(f"\nTotal finite records across 8 days: {total_finite:,} / {total_rec:,} (Duty cycle: {total_finite/total_rec*100:.4f}%)")
print(f"Total contiguous segments: {total_segs}")
