# %%
import imod
from matplotlib import pyplot as plt

from pathlib import Path

# %%

wdir = Path("../../Peelvenen")
imod_python_dir = wdir / "conversion_output"
imod5_dir = wdir / "CONVERSION_IMOD5"

imod_python_name = "peelvenen"
imod5_name = "PEELVENEN"

path_out = wdir / ".." / "comparison_results"
path_out.mkdir(exist_ok=True)

# %%
imod_python_mf6 = imod_python_dir / "modflow6"
imod5_mf6 = imod5_dir / "GWF_1"

head_path_imod5 = imod5_mf6 / "MODELOUTPUT" / "HEAD" / "HEAD.HED"
grb_path_imod5 = imod5_mf6 / "MODELINPUT" / f"{imod5_name}.DIS6.grb"

head_path_python = imod_python_mf6 / f"{imod_python_name}_model" / f"{imod_python_name}_model.hds"
grb_path_python = imod_python_mf6 / f"{imod_python_name}_model" / "dis.dis.grb"

head_imod5 = (
    imod.mf6.open_hds(head_path_imod5, grb_path_imod5).isel(time=-1).compute()
)
head_python = (
    imod.mf6.open_hds(head_path_python, grb_path_python)
    .isel(time=-1)
    .compute()
)

diff_head = head_imod5 - head_python

# %%
lay_nr = 1

fig, ax = plt.subplots()
levels = [-1.2, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.2]
imod.visualize.plot_map(diff_head.sel(layer=lay_nr), colors="RdBu", levels=levels, fig=fig, ax=ax)

ax.set_title(f"dhead iMOD5 - iMOD Python (layer {lay_nr})")

plt.savefig(path_out / f"dhead_iMOD5_iMOD_Python_layer_{lay_nr}.png", dpi=300)

# %%
# Save as IDF

imod.idf.save(path_out / f"dhead", diff_head.sel(layer=lay_nr))

# %%
import numpy as np
fig, ax = plt.subplots()
np.abs(diff_head).plot.hist(yscale="log")
plt.savefig(path_out / "dhead_iMOD5_iMOD_Python_histogram.png", dpi=300)

# %%
