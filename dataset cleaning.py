"""
merge_laptops.py
────────────────
Merges all 7 laptop CSV datasets into one master file,
then consolidates columns with similar/duplicate meanings
into single canonical columns.

104 raw columns  →  49 clean canonical columns
"""

import pandas as pd
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_DIR = Path("Dataset")
OUTPUT_FILE = Path("merged_laptops.csv")

INDEX_COLS = {"unnamed: 0", "indx", "index"}
DROP_EXACT_DUPLICATES = False

# ── File list ─────────────────────────────────────────────────────────────────

FILES = [
    "amazon_laptop_prices_v01.csv",
    "LAPTOP__1_.csv",
    "laptop.csv",
    "laptop_cleaned2.csv",
    "laptops__1_.csv",
    "laptops_cleaned.csv",
    "laptops.csv",
]

# ── Load & tag ────────────────────────────────────────────────────────────────

dfs = []

for fname in FILES:
    path = INPUT_DIR / fname
    if not path.exists():
        print(f"  [SKIP] {fname} not found at {path}")
        continue

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    drop_cols = [c for c in df.columns if c.lower() in INDEX_COLS]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    df["source_file"] = fname
    print(f"  Loaded  {fname:40s}  ->  {df.shape[0]:>5} rows  x  {df.shape[1]-1:>2} cols (excl. source_file)")
    dfs.append(df)

# ── Merge ─────────────────────────────────────────────────────────────────────

print("\nMerging ...")
merged = pd.concat(dfs, axis=0, join="outer", ignore_index=True)

total_before = merged.shape[0]
print(f"  Total rows after concat          : {total_before:,}")
print(f"  Total columns (incl. source_file): {merged.shape[1]}")

# ── Duplicate handling ────────────────────────────────────────────────────────

exact_dupes = merged.duplicated()
print(f"\n  Exact duplicate rows detected    : {exact_dupes.sum():,}")

if DROP_EXACT_DUPLICATES:
    merged.drop_duplicates(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    print(f"  Rows after dropping duplicates   : {merged.shape[0]:,}")
else:
    print("  (kept -- set DROP_EXACT_DUPLICATES=True to remove them)")

# ── Column inventory ──────────────────────────────────────────────────────────

print("\n-- Column inventory (raw) -----------------------------------------------")
coverage = {
    col: merged[col].notna().sum()
    for col in merged.columns
    if col != "source_file"
}
for col, count in sorted(coverage.items(), key=lambda x: -x[1]):
    pct = count / merged.shape[0] * 100
    bar = "#" * int(pct / 5)
    print(f"  {col:<35s}  {count:>6,} / {merged.shape[0]:,}  ({pct:5.1f}%)  {bar}")

# ── Column consolidation ──────────────────────────────────────────────────────
# Map of canonical name -> source columns in priority order.
# combine_first() is used so the first non-null value per row wins.

print("\n-- Consolidating similar columns ----------------------------------------")

CONSOLIDATION = {
    "brand":               ["brand", "Brand", "Company"],
    "model":               ["model", "Model", "Name", "name", "TypeName"],
    "price":               ["price", "Price"],
    "rating":              ["rating", "Rating", "star"],
    "os":                  ["os", "OS", "OpSys", "os_name", "Operating_system"],
    "ram_gb":              ["ram", "Ram", "RAM_GB", "ram_gb"],
    "cpu_brand":           ["cpu_brand", "Processor_brand"],
    "cpu_name":            ["cpu", "cpu_name", "processor", "Processor_name"],
    "cpu_speed_ghz":       ["cpu_speed"],
    "cpu_gen":             ["Generation", "Processor_gen"],
    "cpu_cores":           ["Core", "cpu_core_count", "Core_per_processor"],
    "cpu_threads":         ["Threads", "cpu_thread_count"],
    "cpu_p_cores":         ["cpu_p_cores"],
    "cpu_e_cores":         ["cpu_e_cores"],
    "cpu_lp_e_cores":      ["cpu_lp_e_cores"],
    "cpu_family":          ["cpu_family"],
    "cpu_series":          ["cpu_series"],
    "cpu_model_id":        ["cpu_model"],
    "cpu_suffix":          ["cpu_suffix"],
    "storage_gb":          ["storage_gb", "Storage_capacity_GB", "SSD", "ssd",
                            "primary_storage", "harddisk", "capacity", "Memory"],
    "storage_type":        ["Storage_type", "memory_type"],
    "secondary_storage":   ["secondary_storage", "hdd", "hybrid", "flashstorage"],
    "gpu_brand":           ["gpu_brand", "Graphics_brand"],
    "gpu_name":            ["gpu_name", "Graphics_name", "graphics",
                            "Graphics", "graphics_coprocessor"],
    "gpu_vram_gb":         ["gpu_vram_gb", "Graphics_GB"],
    "gpu_type":            ["gpu_type"],
    "gpu_series":          ["gpu_series"],
    "gpu_model_id":        ["gpu_model"],
    "display_size_inch":   ["display_size_inch", "Display_size_inches",
                            "Inches", "screen_size", "Display"],
    "display_width_px":    ["display_width_px", "Horizontal_pixel", "resolution_width"],
    "display_height_px":   ["display_height_px", "Vertical_pixel", "resolution_height"],
    "display_ppi":         ["ppi"],
    "screen_resolution":   ["screen_resolution"],
    "is_touchscreen":      ["is_touchscreen", "Touch_screen", "touchscreen"],
    "has_ips_panel":       ["has_ips_panel", "ipspanel"],
    "has_retina_display":  ["retinadisplay"],
    "color":               ["color"],
    "special_features":    ["special_features"],
    "device_category":     ["device_category"],
    "ram_type":            ["RAM_type"],
    "warranty":            ["Warranty", "warranty_years"],
    "weight":              ["Weight", "Weight_kg"],
    "rating_review_text":  ["Rating_and_Review"],
    "total_processors":    ["Total_processor"],
    "execution_units":     ["Execution_units"],
    "energy_eff_units":    ["Energy_Efficient_Units"],
    "low_power_cores":     ["Low_Power_Cores"],
    "graphics_integrated": ["Graphics_integreted"],
    "source_file":         ["source_file"],
}

consolidated = pd.DataFrame(index=merged.index)

for canonical, sources in CONSOLIDATION.items():
    available = [c for c in sources if c in merged.columns]
    if not available:
        consolidated[canonical] = pd.NA
        continue
    if len(available) == 1:
        consolidated[canonical] = merged[available[0]]
    else:
        consolidated[canonical] = merged[available[0]]
        for col in available[1:]:
            consolidated[canonical] = consolidated[canonical].combine_first(merged[col])
    filled = consolidated[canonical].notna().sum()
    pct = filled / len(merged) * 100
    print(f"  {canonical:<25s}  {filled:>6,}/{len(merged):,}  ({pct:5.1f}%)  <- {available}")

# ── Save consolidated output ──────────────────────────────────────────────────

consolidated.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved -> {OUTPUT_FILE.resolve()}")
print(f"  Raw shape         : {merged.shape[0]:,} rows x {merged.shape[1]} columns")
print(f"  Consolidated shape: {consolidated.shape[0]:,} rows x {consolidated.shape[1]} columns")
print(f"\n  Final columns ({consolidated.shape[1]}):")
for c in consolidated.columns:
    print(f"    - {c}")