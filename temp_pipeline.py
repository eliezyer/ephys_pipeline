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
import json
# to run catgt from python
import subprocess
# to make the channel map
sys.path.append('utils')
from SGLXMetaToCoords import MetaToCoords
from get_channel_groups import get_channel_groups_with_regions
from generate_xml_with_channel_groups import generate_xml_with_channel_groups
# to parse xml files
import xml.etree.ElementTree as ET

# to run buzcode functions
import matlab.engine

# for running on windows
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
if sessions_to_analyze is None:
    analyze_all_sessions = True
else:
    analyze_all_sessions = False
run_catgt = False 
run_kilosort = False 
run_bombcell = True 
generate_xml = True # generates an xml file for easy data loading into neuroscope and for buzcode
run_buzcode = True 
# these only matter if running buzcode
generate_lfp = True
find_ripples = True
find_SWRs = True # this will use existing ripples if present, otherwise it will detect them
best_SW_channel = 44 # IMPLEMENT AUTO DETECTION OF BEST SW CHANNEL
find_sleep_states = True


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

template_xml_path = 'sample_xml_neuroscope.xml' # path to xml template  to use for generating the new one
#%%
for day in days_to_analyze: # loop through each day/session
    current_day_path = Path(data_basepath, day) # path to the day
    # create supercat folder if it doesnt exist 
    supercat_folder = Path(current_day_path, 'supercat')
    if not os.path.exists(supercat_folder):
        os.mkdir(supercat_folder)

    if analyze_all_sessions: # get all folder names from that day
        sessions_to_analyze = [session for session in os.listdir(current_day_path) if os.path.isdir(Path(current_day_path, session))]
    
    for session in sessions_to_analyze: # loop through each recording # NEED TO IMPLEMENT CATGT CONCAT OPTION
        basepath = Path(current_day_path, session,(session + '_imec0')) # path to the session
        print(f"Processing: {basepath}")

    
        basename = basepath.stem
        os.chdir(basepath.parent)
    
        # searching original binary and meta file
        original_binary_file = list(basepath.glob("*.imec0.ap.bin"))[0] 
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
                '-apfilter=butter,12,300,9000',
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

            # also generate channelmap file for kilosort and xml file generation # CHANGE TO USE SUPERCAT FOLDER
            MetaToCoords(metaFullPath=catgt_meta_file, destFullPath=supercat_folder, outType=5, showPlot=False) # outType 5 is for kilosort json

            if result.returncode !=0:
                raise ValueError("CatGt failed")
            print(f"Successfully ran CatGT")

        # NOT FULLY TESTED YET
        if generate_xml:
            # check if json file exists
            if not os.path.exists(Path(supercat_folder, '*.json')):
                raise FileNotFoundError(f"JSON file {Path(supercat_folder, '*.json')} does not exist - run catgt first")
            # load file ending in chanmap.json in supercat folder 
            json_file_path = list(supercat_folder.glob('*chanmap.json'))[0]
            with open(json_file_path, 'r') as f:
                temp = json.load(f)
            # extract channel positions from json file
            channel_positions = np.array([temp['xc'], temp['yc']]).T

            # find which animal we are dealing with
            if 'NPX1' in session:
                animal_name = 'NPX1'
                region_df = region_df_NPX1
            elif 'NPX2' in session:
                animal_name = 'NPX2'
                region_df = region_df_NPX2
            elif 'NPX3' in session:
                animal_name = 'NPX3'
                region_df = region_df_NPX3

            channel_groups, region_names = get_channel_groups_with_regions(channel_positions, region_df=region_df, 
            x_threshold=x_lim_channel_groups, y_threshold=y_lim_channel_groups)

            generate_xml_with_channel_groups(
                template_xml_path=template_xml_path,
                output_xml_path=Path(supercat_folder, 'neuroscope.xml'),
                channel_groups=channel_groups,
                group_regions=region_names,
            )
            print(f"Successfully generated XML file: {Path(supercat_folder, 'neuroscope.xml')}")

        if run_kilosort:
            
            # get channel groups from xml file to run CAR separately for each channel group
            xml_file_path = Path(supercat_folder, 'neuroscope.xml')
            if not os.path.exists(xml_file_path):
                raise FileNotFoundError(f"XML file {xml_file_path} does not exist - run generate_xml first")

            # parse XML file to extract channel groups for *all* regions
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            region_channels = {}
            brain_regions = root.find('brainRegions')

            # iterate over all region elements inside <brainRegions>
            for region_elem in brain_regions:
                region_name = region_elem.tag 
                channels_element = region_elem.find('channels')
                if channels_element is None or not channels_element.text:
                    # skip regions without a channels entry
                    continue

                # parse space-separated channel numbers into ints
                channels = [int(ch) for ch in channels_element.text.split()]
                region_channels[region_name] = channels

            
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


        # NOT TESTED YET
        if run_buzcode:
            eng = matlab.engine.start_matlab() # start matlab engine
            eng.addpath(eng.genpath('buzcode_functions')) # add matlab functions to path
            if generate_lfp:
            # generate LFP at 1250Hz
                eng.ResampleBinary(original_binary_file,basename+'_1250Hz.lfp',385,1,24) #30000 Hz to 1250 Hz
                print(f"Successfully generated LFP file: {basename+'_1250Hz.lfp'}")
            if find_ripples:
                # check that lfp file exists 
                if not os.path.exists(basename+'_1250Hz.lfp'):
                    raise FileNotFoundError(f"LFP file {basename+'_1250Hz.lfp'} does not exist")
                # check that xml file exists (to get channel numbers)
                xml_file_path = Path(str(basepath.parent), animal_name, 'neuroscope.xml')
                if not os.path.exists(xml_file_path):
                    raise FileNotFoundError(f"XML file {xml_file_path} does not exist")
                
                # parse XML file to extract hippocampal channels
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                
                # list of possible hippocampal region names to search for - Case sensitive!
                hpc_region_names = ['CA1', 'CA2', 'CA3', 'DG', 'HPC', 'Hippocampus']
                
                # find hippocampal channels in brainRegions section
                hpc_channels = []
                brain_regions = root.find('brainRegions')
                if brain_regions is not None:
                    for region_name in hpc_region_names:
                        hpc_element = brain_regions.find(region_name)
                        if hpc_element is not None:
                            channels_element = hpc_element.find('channels')
                            if channels_element is not None and channels_element.text:
                                # parse space-separated channel numbers
                                hpc_channels = [int(ch) for ch in channels_element.text.split()]
                                print(f"Found {len(hpc_channels)} hippocampal channels in region '{region_name}'")
                                break
                
                if not hpc_channels:
                    raise ValueError(f"No hippocampal channels found in XML file. Searched for regions: {hpc_region_names}")

                # get best ripple channel from all hpc channels
                ripple_channel = eng.bz_GetBestRippleChan(basename+'_1250Hz.lfp', hpc_channels)
                print(f"Best ripple channel: {ripple_channel}")

                # find ripples
                ### input parameters ###
                #       basepath       path to a single session to run findRipples on
                #       channel      	Ripple channel to use for detection (0-indexed, a la neuroscope)
                #      'thresholds'  thresholds for ripple beginning/end and peak, in multiples
                #                    of the stdev (default = [2 5]); must be integer values
                #      'durations'   min inter-ripple interval and max ripple duration, in ms
                #                    (default = [30 100]). 
                #      'minDuration' min ripple duration. Keeping this input nomenclature for backwards
                #                    compatibility
                #      'stdev'       reuse previously computed stdev
                #      'show'        plot results (default = 'off')
                #      'noise'       noisy unfiltered channel used to exclude ripple-
                #                    like noise (events also present on this channel are
                #                    discarded)
                #      'passband'    N x 2 matrix of frequencies to filter for ripple detection 
                #                    (default = [130 200])
                #      'EMGThresh'   0-1 threshold of EMG to exclude noise
                #      'saveMat'     logical (default=false) to save in buzcode format

                eng.bz_FindRipples(basename+'_1250Hz.lfp', ripple_channel,
                'thresholds', [2.5, 4], 'durations', [30, 100], 'minDuration', 10, 'noise', 282, 'passband', [100, 250],
                'EMGThresh', 0.8, 'saveMat', True)
                print(f"Successfully extracted ripples from {basename+'_1250Hz.lfp'}")
            
            
            if find_SWRs:
                                # check that lfp file exists 
                if not os.path.exists(basename+'_1250Hz.lfp'):
                    raise FileNotFoundError(f"LFP file {basename+'_1250Hz.lfp'} does not exist")
                # check that xml file exists (to get channel numbers)
                xml_file_path = Path(str(basepath.parent), animal_name, 'neuroscope.xml')
                if not os.path.exists(xml_file_path):
                    raise FileNotFoundError(f"XML file {xml_file_path} does not exist")
                
                # parse XML file to extract hippocampal channels
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                
                # list of possible hippocampal region names to search for - Case sensitive!
                hpc_region_names = ['CA1', 'CA2', 'CA3', 'DG', 'HPC', 'Hippocampus']
                
                # find hippocampal channels in brainRegions section
                hpc_channels = []
                brain_regions = root.find('brainRegions')
                if brain_regions is not None:
                    for region_name in hpc_region_names:
                        hpc_element = brain_regions.find(region_name)
                        if hpc_element is not None:
                            channels_element = hpc_element.find('channels')
                            if channels_element is not None and channels_element.text:
                                # parse space-separated channel numbers
                                hpc_channels = [int(ch) for ch in channels_element.text.split()]
                                print(f"Found {len(hpc_channels)} hippocampal channels in region '{region_name}'")
                                break
                
                if not hpc_channels:
                    raise ValueError(f"No hippocampal channels found in XML file. Searched for regions: {hpc_region_names}")

                # get best ripple channel from all hpc channels
                ripple_channel = eng.bz_GetBestRippleChan(basename+'_1250Hz.lfp', hpc_channels)
                print(f"Best ripple channel: {ripple_channel}")
                print(f"Best SW channel: {best_SW_channel}")

                #  Channels:     an array of two channel IDs following LoadBinary format (base 1)
                #                First RIPPLE channel, and second the SHARP WAVE channel.
                #                Detection in based on these order, please be careful
                #                E.g. [1 11]
                # 
                #  All following inputs are optional:
                # 
                #  basepath:     full path where session is located (default pwd)
                #                e.g. /mnt/Data/buddy140_060813_reo/buddy140_060813_reo
                # 
                #  Epochs:       a list of time stamps [start stop] in seconds demarcating 
                #                epochs in which to detect SWR. If this argument is empty, 
                #                the entire .lfp/.lfp file will be used.               
                # 
                #  saveMat:      logical (default=false) to save in buzcode format
                # 
                #  forceDetect   true or false to force detection (avoid load previous 
                #                detection, default false)
                # 
                #  swBP:         [low high], passband for filtering sharp wave activity
                #  
                #  ripBP:        [low high] passband for filtering around ripple activity
                #  
                #  per_thresswD: a threshold placed upon the sharp wave difference magnitude
                #                of the candidate SWR cluster determined via k-means.
                #  per_thresRip: a threshold placed upon ripple power based upon the non-SWR
                #                cluster determined via k-means.
                #  
                #  WinSize:      window size in milliseconds for non-overlapping detections
                #  
                #  Ns_chk:       sets a window [-Ns_chk, Ns_chk] in seconds around a
                #                candidate SWR detection for estimating local statistics.
                #  
                #  thresSDswD:   [low high], high is a threshold in standard deviations upon
                #                the detected maximum sharp wave difference magnitude, based
                #                upon the local distribution. low is the cutoff for the 
                #                feature around the detected peak for determining the 
                #                duration of the event, also based upon the local
                #                distribution.
                #  
                #  thresSDrip:   [low high], high is a threshold in standard deviations upon
                #                the detected maximum ripple magnitude, based
                #                upon the local distribution. low is the cutoff for the 
                #                feature around the detected peak for determining the 
                #                duration of the event, also based upon the local
                #                distribution.
                #  minIsi:       a threshold setting the minimum time between detections in
                #                seconds.
                #  
                #  minDurSW:     a threshold setting the minimum duration of a localized
                #                sharp-wave event.
                # 
                #  maxDurSW:     a threshold setting the maximum duration of a localized
                #                sharp-wave event.
                # 
                #  minDurRP:     a threshold setting the minimum duration of a localized
                #                ripple event associated with a sharp-wave.
                #      
                #  EVENTFILE:    boolean, a flag that triggers whether or not to write out
                #                an event file corresponding to the detections (for
                #                inspection in neuroscope).
                #  noPrompts     true/false disable any user prompts (default: true)
                # 
                eng.bz_DetectSWR(basepath = basename, Channels = [ripple_channel, best_SW_channel], saveMat=True, EVENTFILE=True)
                print(f"Successfully extracted SWRs from {basename+'_1250Hz.lfp'}")

            if find_sleep_states:
                # check that lfp file exists 
                if not os.path.exists(basename+'_1250Hz.lfp'):
                    raise FileNotFoundError(f"LFP file {basename+'_1250Hz.lfp'} does not exist")
                # check that xml file exists (to get channel numbers)
                xml_file_path = Path(str(basepath.parent), animal_name, 'neuroscope.xml')
                if not os.path.exists(xml_file_path):
                    raise FileNotFoundError(f"XML file {xml_file_path} does not exist")
                
                # list of possible region names to search for - in order of preference! 
                # HPC > CTX > TH for SW and theta channels - Case sensitive!
                region_names = ['CA1', 'CA2', 'CA3', 'DG', 'HPC', 'Hippocampus', 
                'CTX', 'ACC', 'PFC','Cortex', 'TH', 'AM', 'MDT', 'Thalamus']
                
                # find channels in brainRegions section
                region_channels = []
                for region_name in region_names:
                    region_element = brain_regions.find(region_name)
                    if region_element is not None:
                        channels_element = region_element.find('channels')
                        if channels_element is not None and channels_element.text:
                            # parse space-separated channel numbers
                            region_channels = [int(ch) for ch in channels_element.text.split()]

                # find sleep states
                ### input parameters ###
                #    basePath        folder containing .xml and .lfp files.
                #                    basePath and files should be of the form:
                #                    'whateverfolder/recordingName/recordingName'
                #    'savebool'      Default: true. Save anything.
                #    'scoretime'     Window of time to score. Default: [0 Inf] 
                #                    must be continous interval
                #    'ignoretime'    Time intervals winthin scoretime to ignore 
                #                    (for example, opto stimulation or behavior with artifacts)   
                #    'winparms'      [FFT window , smooth window] (Default: [2 15])
                #                    (Note: updated from [10 10] based on bimodaility optimization, 6/17/19)
                #    'Notch60Hz'     Boolean 0 or 1.  Value of 1 will notch out the 57.5-62.5 Hz
                #                    band, default is 0, no notch.  This can be necessary if
                #                    electrical noise.
                #    'NotchUnder3Hz' Boolean 0 or 1.  Value of 1 will notch out the 0-3 Hz
                #                    band, default is 0, no notch.  This can be necessary
                #                    due to poor grounding and low freq movement transients
                #    'NotchHVS'      Boolean 0 or 1.  Value of 1 will notch the 4-10 and 
                #                    12-18 Hz bands for SW detection, default is 0, no 
                #                    notch.  This can be useful in
                #                    recordings with prominent high voltage spindles which
                #                    have prominent ~16hz harmonics
                #    'NotchTheta'    Boolean 0 or 1.  Value of 1 will notch the 4-10 Hz
                #                    band for SW detection, default is 0, no notch.  This 
                #                    can be useful to
                #                    transform the cortical spectrum to approximately
                #                    hippocampal, may also be necessary with High Voltage
                #                    Spindles
                #    'stickytrigger' Implements a "sticky" trigger for SW/EMG threshold 
                #                    crossings: metrics must reach halfway between threshold
                #                    and opposite peak to count as crossing (reduces
                #                    flickering) (default:true)
                #    'SWChannels'    A vector list of channels that may be chosen for SW
                #                    signal
                #    'ThetaChannels' A vector list of channels that may be chosen for Theta
                #                    signal
                #    'MotionSource'  Source for the motion signal used.  'EMGfromLFP' or 
                #                    'Accelerometer'.  Default: EMGfromLFP
                #    'AccelerChans'  Channels devoted to motion/accelerometer.  Needed only
                #                    if the option 'MotionSource" has the value "Accelerometer' 
                #    'rejectChannels' A vector of channels to exclude from the analysis
                #    'saveLFP'       (default:true) to save SleepScoreLFP.lfp.mat file
                #    'noPrompts'     (default:false) an option to not prompt user of things

                eng.SleepScoreMaster(basename, 'ThetaChannels', region_channels, 'SWChannels', region_channels)
                print(f"Successfully extracted sleep states from {basename+'_1250Hz.lfp'}")