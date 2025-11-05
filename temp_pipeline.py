#%% this is the first attempt at creating a pipeline to preprocess the data and extract features


from pathlib import Path
from kilosort import run_kilosort, DEFAULT_SETTINGS
from kilosort.io import load_probe, save_probe
import sys

sys.path.append('utils')
from SGLXMetaToCoords import MetaToCoords


# path wehre the session files are at
basepath = Path('.../2024-06-10_15-45-00.bin.ap')  
baseName = basepath.stem.replace('.bin.ap','')

#%% insert catgt here
# cat gt [ ] tshift the data before sorting
# [ ] in catgt, do the local car, but no filtering required beyond that, kilosort will handle the rest

#%% automatic kilosort
settings = DEFAULT_SETTINGS.copy()
settings['data_dir'] = basepath
settings['n_chan_bin'] = 385

# loading (or creating) the channel map for the probe
# add a if statement later to check if it exists already
# saving the probe for later use / inspection
MetaToCoords( metaFullPath=basepath, outType=5, showPlot=False) # outType 5 is for kilosort json
# loading the probe file
probe_file_name = baseName +'_ks_probe_chanmap.json'
probe_dict = load_probe(probe_file_name)

# running kilosort
ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(settings=settings, probe=probe_dict)

# parameters to play with on kilosort:
# ccg_threshold: splitting merging (should oversplit more for cleaning clusters)
# filtering the data

#%% running bombcell to find good cells

#%% inserting calling buzcode functions
