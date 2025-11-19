#%% this is the first attempt at creating a pipeline to preprocess the data and extract features

# TODO list
# [ ] write a log file for each step
# [ ]

# import
from pathlib import Path
from kilosort import run_kilosort, DEFAULT_SETTINGS
from kilosort.io import load_probe
from datetime import datetime
import sys, os
import numpy as np
import pandas as pd
# to run catgt from python
import subprocess
# to make the channel map
sys.path.append('utils')
from SGLXMetaToCoords import MetaToCoords
from get_channel_groups import get_channel_groups_with_regions
from generate_xml_with_channel_groups import generate_xml_with_channel_groups



# catgt path (this is a fixed location)
# catgt_path = Path(r'C:\Users\Josue Regalado\Documents\EFO_temp_code\utils\J_CatGT-win\CatGT.exe')
# data_basepath = Path(r'C:\Users\Josue Regalado\ephys_temp_data\')

# # [CHANGE ONLY THESE VARIABLES UP HERE]
# days_to_analyze = ['NPX1\11_12_25_pre', 'NPX1\11_13_25_pre', '\NPX1\11_17_25_P1']

# for testing on mac
catgt_path = Path(r'C:/Users/Josue Regalado/Documents/EFO_temp_code/utils/J_CatGT-win/CatGT.exe')
data_basepath = r'/Volumes/memoryShare/Leslie_and_Tim/data/ephys'
days_to_analyze = [r'NPX1/11_13_25_pre',r'NPX1/11_12_25_pre', r'NPX1/11_17_25_P1']

sessions_to_analyze = None # if None, all sessions from that day will be analyzed

run_catgt = False 
run_kilosort = False 
run_bombcell = True 
generate_xml = True # generates an xml file for easy data loading into neuroscope and for buzcode
run_buzcode = True 

#### THIS IS ONLY NEEDED FOR GENERATING THE XML FILE ####
# specify x and y coordinates for at least one site per channel group. If multiple sites are present
# in a channel group, specify coordinates that lie anywhere in the x and y ranges.
# if only one site is present, specify the exact coordinates. List the region name associated 
# with that channel group. The order doesn't matter as long as x, y and region are correctly associated. 
# y coordinates are based off the tip of the probe!
# x coordinates for NP2.0: shank0: 27, 59; shank1: 277, 309; shank2: 527, 559; shank3: 777, 809

# in NPX1 ACC is only single sites, TH are multiple sites. 
region_df_NPX1 = pd.DataFrame({
    'x': [50, 559, 300, 550, 27, 800, 309, 809, 809],
    'y': [200, 3870, 200, 200, 3795, 200, 3600, 3525, 3585],
    'region': ['TH', 'ACC', 'TH', 'TH', 'ACC', 'TH', 'ACC', 'ACC', 'ACC'],
})

region_df_NPX2 = pd.DataFrame({
    'x': [50, 300, 50, 550, 800],
    'y': [200, 3000, 2700, 3000, 3000],
    'region': ['TH', 'ACC', 'ACC', 'ACC', 'ACC'],
})

region_df_NPX3 = pd.DataFrame({
    'x': [50, 300, 800, 550, 800],
    'y': [3000, 3000, 2800, 200, 200],
    'region': ['CA1', 'CA1', 'CA1', 'TH', 'TH'],
})

# how far away sites can be before they are considered to be in different channel groups
y_lim_channel_groups = 106 # allows for 5 empty sites between channel groups
x_lim_channel_groups = 50 # 

# if kilosort hasn't generated the file yet, specify a folder that 
# contains the channel_positions.npy file
alternative_channel_position_path = None 

template_xml_path = 'sample_xml_neuroscope.xml' # path to xml template  to use for generating the new one
#%%
for day in days_to_analyze: # loop through each day
    current_day_path = Path(data_basepath, day) # path to the day

    if sessions_to_analyze is None: # get all folder names from that day
        sessions_to_analyze = [session for session in os.listdir(current_day_path) if os.path.isdir(Path(current_day_path, session))]
    
    for session in sessions_to_analyze: # loop through each session # NEED TO IMPLEMENT CATGT CONCAT OPTION
        basepath = Path(current_day_path, (session + '_imec0')) # path to the session
        print(f"Processing: {basepath}")

    
        basename = basepath.stem
        os.chdir(basepath.parent)
    
        # searching original binary and meta file
        #original_binary_file = list(basepath.glob("*.imec0.ap.bin"))[0] # not sure this is necessary anymore
        #original_meta_file = list(basepath.glob("*.imec0.ap.meta"))[0] # not sure this is necessary anymore
        run_name = basename[:-9] # run name is the core name of folder&files, this is a lazy solution for it

        
        # setting a name for the new.bin file that catgt will create
        catgt_path_save = str(basepath.parent  / basename) + '_catgt'
        # new_binary_file = str(Path(catgt_path_save) /basename) +'_catgt.bin' # not used for now
        catgt_bin_folder = Path(catgt_path_save)
        if run_catgt: 
            # catgt parameters (change here accordingly)
            catgt_params = [
                '-dir='+str(basepath.parent.parent),
                '-run='+run_name,
                '-g=0',
                '-t=0',
                '-prb=0',
                '-prb_fld',
                '-t_miss_ok',
                '-ap',
                '-lf',
                '-apfilter=butter,12,300,9000',
                '-lffilter=butter,12,1,600'
                '-gblcar',
                '-gfix=0.4,0.1,0.02',
                '-dest='+catgt_path_save, # where to save it
                '-no_catgt_fld' # do not create a cat gt folder (we are already making our own)
            ]

            try:
                os.mkdir(catgt_path_save)
                print('Created CatGT directory: '+ catgt_path_save)
            except FileExistsError:
                print('CatGT directory already exists: '+ catgt_path_save)

            print("Attempting to run CatGt")
            # cat_gt_command_str = ' -dir='+str(basepath)+' -run='+run_name+' -g=0 -t=0 -prb_fld -t_miss_ok -ap -probe=0 -apfilter=butter,12,300,9000 -gblcar -gfix-0.4,0.1,0.02 -dest='+catgt_path_save

            #making the command 
            cmd = [str(catgt_path)] + catgt_params #cat_gt_command_str

            # run CatGT
            result = subprocess.run(cmd,capture_output=True,text=True)

            if result.returncode !=0:
                raise ValueError("CatGt failed")

        if run_kilosort:

            # # get the output file of catgt as the file to spike sort
            catgt_binary_file = list(catgt_bin_folder.glob("*.ap.bin"))[0]
            catgt_meta_file = list(catgt_bin_folder.glob("*.ap.meta"))[0] # not sure this is necessary anymore

            # Automatic kilosort settings
            settings = DEFAULT_SETTINGS.copy()
            # settings['data_dir'] = basepath
            settings['filename'] = catgt_binary_file
            settings['n_chan_bin'] = 385

            # (OPTIONAL) parameters to play with on kilosort. Uncomment below to change
            # settings['ccg_threshold']=0.1 # default is 0.25 # ccg_threshold: splitting merging (should oversplit more for cleaning clusters)
            # settings['highpass_cutoff']=500 # in Hz, default is 300 # filtering the data for spike sorting  


            # loading (or creating) the channel map for the probe
            # add a if statement later to check if it exists already
            # saving the probe for later use / inspection
            # FIX THIS LATER

            meta_file_name = catgt_meta_file
            MetaToCoords( metaFullPath=meta_file_name, outType=5, showPlot=False) # outType 5 is for kilosort json
            # loading the probe file 
            probe_file_name = list(catgt_bin_folder.glob('*_ks_probe_chanmap.json'))[0] 
            probe_dict = load_probe(catgt_bin_folder / probe_file_name)

            # setting kilosort folder name
            date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            ks_folder_save_name = catgt_bin_folder / Path('kilosort4_'+date_time)
            # running kilosort
            ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
                run_kilosort(settings=settings, probe=probe_dict,results_dir=ks_folder_save_name)

        if run_bombcell:
            print("not implemented yet")

        # NOT FULLY FUNCTIONAL YET
        if generate_xml:
            if alternative_channel_position_path is not None:
                channel_positions = np.load(alternative_channel_position_path)
            else:
                channel_positions = np.load(ks_folder_save_name / Path('channel_positions.npy'))

            # find folder that has kilosort in it 
            kilosort_folder = list(catgt_bin_folder.parent.glob('kilosort*'))[0]

            channel_groups, region_names = get_channel_groups_with_regions(channel_positions, region_df=region_df, x_threshold=50, y_threshold=50)

            generate_xml_with_channel_groups(
                template_xml_path=template_xml_path,
                output_xml_path='NPX1_neuroscope_with_channel_groups.xml',
                channel_groups=channel_groups,
                group_regions=region_names,
            )
        #%% inserting calling buzcode functions
        if run_buzcode:
            print("not implemented yet")

        sessions_to_analyze = None # reset the sessions to analyze for the next day