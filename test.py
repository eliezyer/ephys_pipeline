#%%
import json

path = '/Volumes/memoryShare/Leslie_and_Tim/data/ephys/NPX3/11_13_25_pre/NPX3_11_13_25_offline2_CA_TH_g0/NPX3_11_13_25_offline2_CA_TH_g0_imec0_catgt/NPX3_11_13_25_offline2_CA_TH_g0_tcat.imec0.ap_ks_probe_chanmap.json'

with open(path, 'r') as f:
    test = json.load(f)
# %%
test['xc']
# %%
channel_coordinates = np.array([test['xc'], test['yc']]).T
test2 = np.load('/Volumes/memoryShare/Leslie_and_Tim/data/ephys/NPX3/11_13_25_pre/NPX3_11_13_25_offline2_CA_TH_g0/NPX3_11_13_25_offline2_CA_TH_g0_imec0_catgt/kilosort4_20251115_152023/channel_positions.npy')
# %%
import numpy as np
# %%
np.allclose(channel_coordinates, test2)
# %%
