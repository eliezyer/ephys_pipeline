#%% this is the first attempt at creating a pipeline to preprocess the data and extract features
# TODO list
# [ ] write a log file for each step
# [ ]

# import

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir) # make sure we start out from the script directory
sys.path.append('utils')
sys.path.append('utils\purrito')
from pathlib import Path
from purrito import CatGt_wrapper
from kilosort import run_kilosort, DEFAULT_SETTINGS
from kilosort.io import load_probe
from datetime import datetime
import csv
import numpy as np
import pandas as pd
import json

from SGLXMetaToCoords import MetaToCoords
from get_channel_groups import get_channel_groups_with_regions
from generate_xml_with_channel_groups import generate_xml_with_channel_groups
from get_info_from_xml import get_all_channel_groups_from_xml
from get_info_from_xml import get_subset_channels_from_xml
from concat_event_times import concat_event_times
# to run buzcode functions
import matlab.engine

# for running on windows
# catgt path (this is a fixed location)
catgt_path = Path(r'C:\Users\Josue Regalado\Documents\EFO_temp_code\utils\J_CatGT-win\CatGT.exe')
data_basepath = Path(r'C:\Users\Josue Regalado\ephys_temp_data')

# [CHANGE ONLY THESE VARIABLES UP HERE]
days_to_analyze = [r'NPX3\11_17_25_P1']

# for testing on mac
# catgt_path = Path(r'C:/Users/Josue Regalado/Documents/EFO_temp_code/utils/J_CatGT-win/CatGT.exe')
# data_basepath = r'/Volumes/memoryShare/Leslie_and_Tim/data/ephys'
# days_to_analyze = [r'NPX1/11_13_25_pre',r'NPX1/11_12_25_pre', r'NPX1/11_17_25_P1']

sessions_to_analyze = None # if None, all sessions from that day will be analyzed and concatenated
if sessions_to_analyze is None:
    analyze_all_sessions = True
else:
    analyze_all_sessions = False
run_catgt = True 
generate_xml = True # generates an xml file for easy data loading into neuroscope and for buzcode
spikesort = True 
car_separately = True # if True, will CAR separately for each channel group
sort_seperatly = True # if True, will run kilosort separately for each channel group
run_bombcell = False 
run_buzcode = False # generates LFP, finds sleep states and ripples, SWRs (only for HPC)
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
# below is for a NPX1 test recording (10_27_2025)
# region_df_NPX1 = pd.DataFrame({
#     'x': [550, 50, 300, 800, 50, 300],
#     'y': [3600, 100, 100, 3600, 3700, 3700],
#     'region': ['ACC', 'TH', 'TH', 'ACC', 'ACC', 'ACC'],
# })
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

# template to create new xml files
template_xml_path = Path(script_dir, 'utils', 'sample_xml_neuroscope.xml')
#%%
for day in days_to_analyze: # loop through each day/session
    current_day_path = Path(data_basepath, day) # path to the day
    # set a supercat folder (concatenated files) and create it
    supercat_folder = Path(current_day_path, 'finalcat')
    if not os.path.exists(supercat_folder):
        os.mkdir(supercat_folder)

    if analyze_all_sessions: # get all folder names from that day
        sessions_to_analyze = [session for session in os.listdir(current_day_path) if os.path.isdir(Path(current_day_path, session))]

    # exluce finalcat folder from sessions_to_analyze
    sessions_to_analyze = np.array(sessions_to_analyze)
    sessions_to_analyze = np.delete(sessions_to_analyze, np.where(sessions_to_analyze == 'finalcat')[0])

    Recording_start_times = []
    # sort sessions in chronological order 
    for tmp_sess in sessions_to_analyze:
        tmp_meta_file = list(Path(current_day_path, tmp_sess, tmp_sess +'_imec0').glob('*ap.meta'))[0]
        tmp_meta_file = readMeta(tmp_meta_file)
        Recording_start_times.append(tmp_meta_file['fileCreateTime'])
    sessions_to_analyze = sessions_to_analyze[np.argsort(Recording_start_times)]
    Recording_start_times = np.sort(Recording_start_times) # also sort the times because we will save them in a csv in case we need them later
    dir_runs = []
    for_cleanup = []
    #%% running catgt on each session individually, then supercat to concatenate everything
    if run_catgt:
        # for loop, each session run
        for i, session in enumerate(sessions_to_analyze): # loop through each recording 
            run_name = session[:-3] # removes _g0

            os.chdir(Path(current_day_path, session, session +'_imec0')) # change cwd so that the CatGT log is saved there
            print('CatGT is processing: ' + run_name)
            for_cleanup.append(Path(current_day_path, session, session +'_imec0'))
            if i==0:
            # intializing CatGt wrapper
                catgt = CatGt_wrapper(
                    catgt_path=catgt_path, # mandatory path to CatGt executable
                    basepath=current_day_path, # mandatory basepath where data is located
                    run_name=run_name
                )
                # setting input and streams
                catgt.set_input(prb=0, prb_fld=True) # setting input probe and probe field
                catgt.set_streams(ap=True,ob=True,obx=0) #obx has to be set if processing onebox
                #catgt.set_streams(ap=True,ob=False)
                catgt.set_options({'t_miss_ok':True,# setting other options
                                'no_catgt_fld':True,
                                'gfix':'0.4,0.1,0.02',
                                'pass1_force_ni_ob_bin':True}) #pass1_force_ni_ob_bin is required to force make a tcat ob file to then concatenate everything
                # running catgt for each session separately
                catgt.run()
            else:
                catgt_new = catgt.clone(
                    basepath=current_day_path,
                    run_name=run_name
                )
                catgt_new.run()

            # for supercat
            dir_runs.append(
            {
            'dir': Path(current_day_path),
            'run_ga': session
            })

        #%% running supercat to concatenate all sessions together
        # do we need to include trim_edges = true or is it true by defaulT????
        os.chdir(supercat_folder) # so that catgt log is saved here
        print('SuperCatting...')
        catgt_sc = CatGt_wrapper(catgt_path=catgt_path,basepath=current_day_path) # basepath here doesn't matter, it is just required
        catgt_sc.set_streams(ap=True,ob=True,obx=0) #obx has to be set if processing onebox (required)
        #catgt_sc.set_streams(ap=True,ob=False) #obx has to be set if processing onebox (required)
        catgt_sc.set_input(prb=0, prb_fld=True) # setting input probe and probe field
        catgt_sc.set_supercat(runs=dir_runs,dest=supercat_folder)
        catgt_sc.run()

        #%% extra processing can be added here: e.g. extracting the TTLs from the obx.bin channels

        # TODO: [ ] test extraction on obx files alone, on multiple channels
        # NOT TESTED YET
        # run_name_finalcat = ... #this needs completion
        # catgt_ob = CatGt_wrapper(catgt_path=catgt_path,basepath=supercat_folder,run_name=run_name_finalcat) # basepath here doesn't matter, it is just required

        # catgt_ob.set_streams(ob=True,obx=0) #obx has to be set if processing onebox (required)
        # catgt_ob.set_options({'xd:1,0,-1,0,0'}) # for bit index 0
        # catgt_ob.run()
    
        # also generate channelmap file for kilosort and xml file generation
        supercat_folder = list(supercat_folder.glob("supercat*"))[0] # go into folder created in finalcat
        catgt_meta_file = list(supercat_folder.glob('*ap.meta'))[0]
        _ = MetaToCoords(metaFullPath=Path(catgt_meta_file), destFullPath=str(list(supercat_folder.glob("supercat*"))[0]), outType=5, showPlot=False) # outType 5 is for kilosort json

        # remove catgt files for individual sessions after supercat
        for s in for_cleanup:   
            temp_bin = list(s.glob("*tcat.imec0.ap.bin"))[0]
            temp_meta = list(s.glob("*tcat.imec0.ap.meta"))[0] 
            temp_catlog = list(s.glob("CatGT.log"))[0] 
            temp_sync_file = list(s.glob("*.ap.xd*"))[0] 
            os.remove(temp_bin)
            os.remove(temp_meta)
            os.remove(temp_catlog)
            os.remove(temp_sync_file)
        
    if generate_xml:
        try:
            # load file ending in chanmap.json in supercat folder 
            json_file_path = list(supercat_folder.glob('*chanmap.json'))[0]
        except:
            supercat_folder = list(supercat_folder.glob("*g0"))[0] # go into folder created in finalcat
            # load file ending in chanmap.json in supercat folder 
            json_file_path = list(supercat_folder.glob('*chanmap.json'))[0]

        with open(json_file_path, 'r') as f:
            temp = json.load(f)
        # extract channel positions from json file
        channel_positions = np.array([temp['xc'], temp['yc']]).T
        # find which animal we are dealing with
        if 'NPX1' in day:
            animal_name = 'NPX1'
            region_df = region_df_NPX1
        elif 'NPX2' in day:
            animal_name = 'NPX2'
            region_df = region_df_NPX2
        elif 'NPX3' in day:
            animal_name = 'NPX3'
            region_df = region_df_NPX3

        channel_groups, region_names = get_channel_groups_with_regions(channel_positions, region_df=region_df, 
        x_threshold=x_lim_channel_groups, y_threshold=y_lim_channel_groups)

        generate_xml_with_channel_groups(
            template_xml_path=template_xml_path,
            output_xml_path=Path(supercat_folder, 'neuroscope.xml'),
            channel_groups=channel_groups,
            group_regions=region_names,
            channel_positions=channel_positions,
        )
        print(f"Successfully generated XML file: {Path(supercat_folder, 'neuroscope.xml')}")

    if spikesort:
        if car_separately or sort_seperatly: # don't need channel groups if everything is run together 
            try:
                xml_file_path = Path(supercat_folder, 'neuroscope.xml')
                region_channels = get_all_channel_groups_from_xml(xml_file_path) # returns a dict with region names associated with channels
            except: 
                supercat_folder = list(supercat_folder.glob("*g0"))[0] # go into folder created in finalcat
                xml_file_path = Path(supercat_folder, 'neuroscope.xml')
                region_channels = get_all_channel_groups_from_xml(xml_file_path) # returns a dict with region names associated with channels
            region_channels_list = list(region_channels.values()) # just need a list for kilosort
            region_names = list(region_channels.keys())
        else:
            region_channels_list = None
        
        # get the output file of catgt as the file to spike sort
        catgt_binary_file = list(supercat_folder.glob("*.ap.bin"))[0]
        catgt_meta_file = list(supercat_folder.glob("*.ap.meta"))[0] # not sure this is necessary anymore

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

        # loading the probe file 
        probe_file_name = list(supercat_folder.glob('*_ks_probe_chanmap.json'))[0] 
        probe_dict = load_probe(supercat_folder / probe_file_name)
        if sort_seperatly: # run kilosort separately for each channel group
            for i in range(len(region_channels_list)):
                # exclude all channel groups except the current one 
                bad_channels = region_channels_list[:i] + region_channels_list[i+1:]
                # make sure its a 1d list 
                bad_channels = [item for sublist in bad_channels for item in sublist]

                # setting kilosort folder name
                date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                ks_folder_save_name = supercat_folder / Path('kilosort4_'+date_time + '_' + region_names[i])
                # running kilosort
                ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
                    run_kilosort(settings=settings, probe=probe_dict,results_dir=ks_folder_save_name, bad_channels = bad_channels)
        else: # run kilosort on all channels together but CAR separately for each channel group
            # setting kilosort folder name
                date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                ks_folder_save_name = supercat_folder / Path('kilosort4_'+date_time)
                # running kilosort
                ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
                    run_kilosort(settings=settings, probe=probe_dict,results_dir=ks_folder_save_name, channel_groups=region_channels_list)
    
    #%% TODO: implement TPrime after kilosort processing

    #%%
    if run_bombcell:
        print("not implemented yet")


    # NOT TESTED YET
    if run_buzcode:
        eng = matlab.engine.start_matlab() # start matlab engine
        eng.addpath(eng.genpath('utils/buzcode/')) # add matlab functions to path
        try:
            catgt_binary_file = list(supercat_folder.glob("*.ap.bin"))[0]
        except:
            supercat_folder = list(supercat_folder.glob("*g0"))[0] # go into folder created in finalcat
            catgt_binary_file = list(supercat_folder.glob("*tcat.imec0.ap.bin"))[0] # supercat bin file
        basename = Path(supercat_folder, supercat_folder.stem[:-3]) # supercat folder + run name
        if generate_lfp:
        # generate LFP at 1250Hz
            eng.ResampleBinary(catgt_binary_file, basename+'_1250Hz.lfp',385,1,24, nargout=0) #30000 Hz to 1250 Hz
            print(f"Successfully generated LFP file: {basename+'_1250Hz.lfp'}")
        # check if lfp file exists because it is needed for the rest of the functions

        if not os.path.exists(basename+'_1250Hz.lfp'):
            raise FileNotFoundError(f"LFP file {basename+'_1250Hz.lfp'} does not exist")
        # load lfp 
        lfp = eng.bz_GetLFP( 'all', 'basepath', str(supercat_folder), 'noPrompts', True)
        
        if find_ripples:
            # check that xml file exists (to get channel numbers)
            xml_file_path = Path(supercat_folder, 'neuroscope.xml')
            if not os.path.exists(xml_file_path):
                raise FileNotFoundError(f"XML file {xml_file_path} does not exist")            

            # get hpc channels from xml file
            hpc_channels = get_subset_channels_from_xml(xml_file_path, region='hpc')
            # get best ripple channel from all hpc channels (0-indexed)
            ripple_channel = eng.bz_GetBestRippleChan(lfp, np.array(hpc_channels) + 1, 1250.0) # +1 because matlab is stupid
            print(f"Best ripple channel: {int(ripple_channel)}")

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
            xml_file_path = Path(supercat_folder, 'neuroscope.xml')
            if not os.path.exists(xml_file_path):
                raise FileNotFoundError(f"XML file {xml_file_path} does not exist")            
            hpc_channels = get_subset_channels_from_xml(xml_file_path, region='hpc')

            ripple_channel = eng.bz_GetBestRippleChan(lfp, np.array(hpc_channels) + 1, 1250.0)

            SW_channel = eng.bz_GetBestSharpWaveChan(lfp, 385, 1250.0, ripple_channel, channel_positions)
            print(f"Best ripple channel: {int(ripple_channel)}")
            print(f"Best SW channel: {best_SW_channel}")
            # TESTED UNTIL HERE
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
            eng.bz_DetectSWR(basepath = basename, Channels = [ripple_channel, SW_channel], saveMat=True, EVENTFILE=True)
            print(f"Successfully extracted SWRs from {basename+'_1250Hz.lfp'}")

        if find_sleep_states:
            # check that lfp file exists 
            if not os.path.exists(basename+'_1250Hz.lfp'):
                raise FileNotFoundError(f"LFP file {basename+'_1250Hz.lfp'} does not exist")
            # check that xml file exists (to get channel numbers)
            xml_file_path = Path(str(basepath.parent), animal_name, 'neuroscope.xml')

            # finds channels for HPC > CTX > TH for SW and theta channels 
            region_channels = get_subset_channels_from_xml(xml_file_path, region='all')
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