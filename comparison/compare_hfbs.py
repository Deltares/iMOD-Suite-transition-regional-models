# %%
import pandas as pd
import numpy as np

from pathlib import Path
import imod
import xugrid as xu
import xarray as xr

import matplotlib.pyplot as plt

# %%
# Path management

wdir = Path("../../Peelvenen")
imod_python_dir = wdir / "conversion_output"
imod5_dir = wdir / "CONVERSION_IMOD5"

imod_python_name = "peelvenen"
imod5_name = "PEELVENEN"

path_out = wdir / ".." / "comparison_results"
path_out.mkdir(exist_ok=True)

path_python = imod_python_dir / rf"modflow6/{imod_python_name}_model\hfb_merged\hfb.dat"
path_imod5 = imod5_dir / rf"GWF_1\MODELINPUT\HFB6\HFB_T1.ARR"

# %%
# Load hfbs
settings = dict(delimiter= '\s+', index_col=False)
python_cols = ["layer1", "row1", "col1", "layer2", "row2", "col2", "hydraulic_characteristic"]
imod5_cols = python_cols + ["sys"]
hfb_python = pd.read_csv(path_python, names=python_cols, skiprows=1, **settings)
hfb_imod5 = pd.read_csv(path_imod5, names=imod5_cols,  **settings)
hfb_imod5 = hfb_imod5.drop(columns="sys")

hfb_python["software"] = "python"
hfb_imod5["software"] = "imod5"
hfb_python["id"] = -1
hfb_imod5["id"] = 1

# %%
index_cols = ["layer1", "row1", "col1", "layer2", "row2", "col2"]
df_all = pd.concat([hfb_python, hfb_imod5])
df_diff = df_all.drop_duplicates(keep=False, subset=index_cols)
df_diff.to_csv(path_out / "hfb_different_cells.csv", index=False, sep=";")

# %%
path_grb = imod_python_dir / rf"modflow6/{imod_python_name}_model/dis.dis.grb"
grb_content = imod.mf6.out.read_grb(path_grb)
idomain = grb_content["idomain"]
idomain = idomain.where(idomain > -2147483648, 0)

idomain_ugrid = xu.UgridDataArray.from_structured(idomain)

# %%
# Convert hfb cellids to face ids

cellid1 = df_all[index_cols[1:3]].values.astype(int)
cellid2 = df_all[index_cols[4:]].values.astype(int)

face_id1 = np.ravel_multi_index(cellid1.T - 1, idomain.shape[1:])
face_id2 = np.ravel_multi_index(cellid2.T - 1, idomain.shape[1:])

# Find the shared edge between the two faces for each HFB row.
grid = idomain_ugrid.ugrid.grid
fe_connectivity = grid.face_edge_connectivity
edges_face1 = fe_connectivity[face_id1]
edges_face2 = fe_connectivity[face_id2]

edge_intersections = [
	np.intersect1d(e1[e1 >= 0], e2[e2 >= 0])
	for e1, e2 in zip(edges_face1, edges_face2)
]

df_all["shared_edge_id"] = np.ravel(edge_intersections)

#%% 
# Create empty UgridDataArray with the same shape as the ugrid, but values on the edges
edge_dim = grid.edge_dimension
layers = idomain.coords["layer"]
new_data = xr.DataArray(np.full((layers.size, grid.sizes[edge_dim]), np.nan), dims=[layers.name, edge_dim], coords={layers.name: layers})
uda = xu.UgridDataArray(new_data, grid=grid)

# %% Assign value to the edge. HFB values cannot be assigned to edges inbetween
# layers, so taking layer1 here is suffices.
cols_to_group = ["layer1", "shared_edge_id", "id"]
grouped = df_all[cols_to_group].groupby(cols_to_group[:-1]).sum()

hfb_layers = grouped.index.get_level_values("layer1").unique()

for layer in hfb_layers:
    group = grouped.loc[layer]
    uda.loc[{layers.name: layer, edge_dim: group.index}] = group["id"].values

# %%
from imod.visualize.common import _cmapnorm_from_colorslevels

cmap = "Blues"
levels = [-1.1, -0.1, 0.1, 1.1]

cmap, norm = _cmapnorm_from_colorslevels(cmap, levels=levels)

# Center labels between level boundaries.
new_labels = ["python", "both", "iMOD5"]
tick_centers = [(left + right) / 2 for left, right in zip(levels[:-1], levels[1:])]

fig, axes = plt.subplots(
    nrows=5, ncols=5, figsize=(16, 15), sharex=True, sharey=True
)

for i, (layer, ax) in enumerate(zip(hfb_layers[:25], axes.flatten())):
    plot_colorbar = False
    if i % 5 == 4:
        plot_colorbar=True

    plot = uda.sel(layer=layer).ugrid.plot(cmap=cmap, norm=norm, ax=ax, add_colorbar=plot_colorbar)
    if plot_colorbar:
        plot.colorbar.set_ticks(tick_centers)
        plot.colorbar.set_ticklabels(new_labels)
        plot.colorbar.set_label("HFB source")

fig.savefig(path_out / "hfb_comparison.png", dpi=300, bbox_inches="tight")

# %%
