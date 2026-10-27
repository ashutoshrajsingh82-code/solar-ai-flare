import json
from pathlib import Path

manifest_path = Path(__file__).resolve().parents[1] / "reports" / "phase4p_expanded_dataset_manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    m = json.load(f)

print(f"Total Products Indexed       : {m['total_products_indexed']}")
print(f"Eligible for Forecasting     : {m['eligible_for_forecasting']}")
print(f"Excluded Products            : {m['excluded_products']}")
print(f"Duplicate Counterpart Groups : {m['duplicate_counterpart_groups']}")
print(f"Overlapping Conflicts        : {m['overlapping_interval_conflicts']}")

print("\nEligible SoLEXS SDD2 Lightcurves:")
total_finite = 0
total_rec = 0
for i in m["inventory"]:
    if i["eligibility_decision"] == "ELIGIBLE":
        total_finite += i["finite_records"]
        total_rec += i["total_records"]
        print(f"  - {i['observation_date']}: records={i['finite_records']}/{i['total_records']}, "
              f"duty={i['duty_cycle_pct']}%, segs={i['num_contiguous_segments']}, "
              f"mean={i['mean_count_rate']} c/s, max={i['max_count_rate']} c/s, "
              f"start={i['start_time_utc']}, end={i['end_time_utc']}, sha={i['member_sha256']}")

print(f"\nTotal finite records across 6 days: {total_finite:,} / {total_rec:,} (Overall duty cycle: {total_finite/total_rec*100:.2f}%)")

print("\nCounterpart Collision Groups:")
for g in m["duplicate_groups"]:
    print(f"  - Type: {g.get('type')}")
    print(f"    Identifier: {g.get('base_identifier')}")
    print(f"    Versions: {g.get('versions_found')}")
    print(f"    Packages: {g.get('packages')}")
    print(f"    Blocker: {g.get('blocker_id')}")

print("\nExcluded Product Summary by Reason:")
excl_counts = {}
for i in m["inventory"]:
    if i["eligibility_decision"] != "ELIGIBLE":
        code = i.get("exclusion_code", "UNKNOWN")
        excl_counts[code] = excl_counts.get(code, 0) + 1
for code, count in sorted(excl_counts.items()):
    print(f"  - {code}: {count} products")
