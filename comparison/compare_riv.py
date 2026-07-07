# %%
import pandas as pd

from pathlib import Path

sys = 3

wdir = Path("../../Peelvenen")

imod_python_dir = wdir / "conversion_output"
imod5_dir = wdir / "CONVERSION_IMOD5"
out_dir = wdir / ".." / "comparison_results"
modelname = "peelvenen"

# Path management
path_riv_python = imod_python_dir / rf"modflow6/{modelname}_model\riv-{sys}riv\riv-0.dat"
path_drn_python = imod_python_dir / rf"modflow6/{modelname}_model\riv-{sys}drn\drn-0.dat"
path_riv_imod5 = imod5_dir / rf"GWF_1\MODELINPUT\RIV6\SYS{sys}\RIV_T1.ARR"

out_dir.mkdir(exist_ok=True, parents=True)

# Load RIVs
settings = dict(delimiter= '\s+', index_col=False)
drn_cols = ["layer", "row", "col", "stage", "conductance"]
riv_cols = drn_cols + ["bottom_elevation"]
riv_cols_imod5 = riv_cols + ["sys", "io"]
drn_python = pd.read_csv(path_drn_python, names=drn_cols, skiprows=1, **settings)
riv_python = pd.read_csv(path_riv_python, names=riv_cols, skiprows=1, **settings)
riv_imod5 = pd.read_csv(path_riv_imod5, names=riv_cols_imod5, skipfooter=12,  **settings).drop(columns=["sys", "io"])

# Combine riv and drn into single riv, these were split up for the infiltration factor.
drn_python["bottom_elevation"] = drn_python["stage"]
riv_python_total = pd.concat([drn_python, riv_python]).round(3)

riv_python_total["software"] = "python"
riv_imod5["software"] = "imod5"

index_cols = ["layer", "row", "col"]
df_diff = pd.concat([riv_python_total, riv_imod5]).drop_duplicates(keep=False, subset=index_cols)
# Reorder columns and save
df_diff = df_diff[["software", "layer", "col", "row", "stage", "conductance", "bottom_elevation"]]
df_diff.to_csv(out_dir / "riv_different_cells.csv", index=False, sep=";")

# %%
# Groupby layer, row, col
methods= {"conductance": "sum", "stage": "first", "bottom_elevation": "min"}
varnames = ["conductance", "stage", "bottom_elevation"]
riv_python_agg = riv_python_total.groupby(index_cols).agg(methods)
riv_imod5_agg = riv_imod5.groupby(index_cols).agg(methods)
riv_diff_agg = riv_python_agg.sort_index() - riv_imod5_agg.sort_index()
# Filter out rows where all differences are NaN (due to missing cells in one of the datasets)
riv_diff_agg = riv_diff_agg.dropna(subset=varnames).sort_values("conductance")
# Filter out rows where all differences are zero
to_drop = (riv_diff_agg[varnames] == 0).all(axis=1)
riv_diff_agg = riv_diff_agg.loc[~to_drop]
riv_diff_agg.to_csv(out_dir / "riv_value_differences.csv", sep=";")

# %%
