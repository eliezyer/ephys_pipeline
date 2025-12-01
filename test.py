#%% 
import matlab.engine
import sys, os
import json
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append('utils')
from SGLXMetaToCoords import MetaToCoords
from SGLXMetaToCoords import readMeta
from get_channel_groups import get_channel_groups_with_regions
from generate_xml_with_channel_groups import generate_xml_with_channel_groups
from get_info_from_xml import get_all_channel_groups_from_xml
from get_info_from_xml import get_subset_channels_from_xml
from concat_event_times import concat_event_times
import xml.etree.ElementTree as ET
#%%
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir) # make sure we start out from the script directory

eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('utils/buzcode/'))
#%%
bin_file = '/Users/timeilers/Desktop/NPX3_11_13_25_offline2_CA_TH_g0_imec0/NPX3_11_13_25_offline2_CA_TH_g0_t0.imec0.ap.bin'
basename = '/Users/timeilers/Desktop/NPX3_11_13_25_offline2_CA_TH_g0_imec0/NPX3_11_13_25_offline2_CA_TH_g0_t0'
eng.ResampleBinary(bin_file,basename+'_2500Hz.lfp',385,1,12,nargout=0) #30000 Hz to 2500 Hz

#%%
_ = MetaToCoords(metaFullPath=Path('/Users/timeilers/Desktop/NPX3_11_13_25_offline2_CA_TH_g0_imec0/NPX3_11_13_25_offline2_CA_TH_g0_t0.imec0.ap.meta'), destFullPath=str('/Users/timeilers/Desktop/NPX3_11_13_25_offline2_CA_TH_g0_imec0/'), outType=5, showPlot=False) # outType 5 is for kilosort json

#%%
y_lim_channel_groups = 106 # allows for 5 empty sites between channel groups
x_lim_channel_groups = 50 # 
template_xml_path = Path(script_dir, 'utils', 'sample_xml_neuroscope.xml')
json_file_path = '/Users/timeilers/Desktop/NPX3_11_13_25_offline2_CA_TH_g0_imec0/NPX3_11_13_25_offline2_CA_TH_g0_t0.imec0.ap_ks_probe_chanmap.json'
with open(json_file_path, 'r') as f:
    temp = json.load(f)
# extract channel positions from json file
channel_positions = np.array([temp['xc'], temp['yc']]).T
# find which animal we are dealing with
region_df_NPX3 = pd.DataFrame({
    'x': [50, 300, 800, 550, 800],
    'y': [3000, 3000, 2800, 200, 200],
    'region': ['CA1', 'CA1', 'CA1', 'TH', 'TH'],
})
channel_groups, region_names = get_channel_groups_with_regions(channel_positions, region_df=region_df_NPX3, 
x_threshold=x_lim_channel_groups, y_threshold=y_lim_channel_groups)

generate_xml_with_channel_groups(
    template_xml_path=template_xml_path,
    output_xml_path=Path('/Users/timeilers/Desktop/NPX3_11_13_25_offline2_CA_TH_g0_imec0', 'neuroscope.xml'),
    channel_groups=channel_groups,
    group_regions=region_names,
    channel_positions=channel_positions,
)

# %%
# parse XML file to extract channel positions
xml_file_path = '/Users/timeilers/Desktop/NPX3_11_13_25_offline2_CA_TH_g0_imec0/neuroscope.xml'

print(channel_positions)

with open(json_file_path, 'r') as f:
            temp = json.load(f)
        # extract channel positions from json file
channel_positions_json = np.array([temp['xc'], temp['yc']]).T

#%% 

lfp = eng.bz_GetLFP( 'all', 'basepath', '/Users/timeilers/Desktop/NPX3_11_13_25_offline2_CA_TH_g0_imec0', 'noPrompts', True)

hpc_channels = get_subset_channels_from_xml(xml_file_path, region='hpc')    
# get best ripple channel from all hpc channels
ripple_channel = eng.bz_GetBestRippleChan(lfp, np.array(hpc_channels) + 1, 1250.0)
print(f"Best ripple channel: {int(ripple_channel)}")

# %%