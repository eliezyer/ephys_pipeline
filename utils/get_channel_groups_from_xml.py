import os 
import xml.etree.ElementTree as ET


def get_all_channel_groups_from_xml(xml_file_path):
    '''
    Extracts ALL channel groups from the XML file
    
    Args:
        xml_file_path: Path to the XML file
    
    Returns:
        region_channels: Dictionary with region names as keys and lists of channel numbers as values
    '''

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

    return region_channels

def get_subset_channels_from_xml(xml_file_path, region):
    '''
    return the channels for the first match found in region_names list. e.g. if region is 
    'hpc', and CA1 is in the XML, returns channels only for CA1. 
    
    Args:
        xml_file_path: Path to the XML file
        region: determines which region_names list to use
    
    Returns:
        region_channels: List of channel numbers
    '''
    if not os.path.exists(xml_file_path):
        raise FileNotFoundError(f"XML file {xml_file_path} does not exist")
    
    # parse XML file to extract hippocampal channels
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    if region == 'hpc':
        region_names = ['CA1', 'CA2', 'CA3', 'DG', 'HPC', 'Hippocampus']
    elif region == 'ctx':
        region_names = ['CTX', 'ACC', 'PFC','Cortex']
    elif region == 'th':
        region_names = ['TH', 'AM', 'MDT', 'Thalamus']
    elif region == 'all':
        region_names = ['CA1', 'CA2', 'CA3', 'DG', 'HPC', 'Hippocampus', 
        'CTX', 'ACC', 'PFC','Cortex', 'TH', 'AM', 'MDT', 'Thalamus']
    else:
        raise ValueError(f"Invalid region: {region}")
    
    # find channels in brainRegions section
    region_channels = []
    brain_regions = root.find('brainRegions')
    if brain_regions is not None:
        for region_name in region_names:
            region_element = brain_regions.find(region_name)
            if region_element is not None:
                channels_element = region_element.find('channels')
                if channels_element is not None and channels_element.text:
                    # parse space-separated channel numbers
                    region_channels = [int(ch) for ch in channels_element.text.split()]
                    print(f"Found {len(region_channels)} channels in region '{region_name}'")
                    break
    

    if not region_channels:
        raise ValueError(f"No channels found in XML file. Searched for regions: {region_names}")
    return region_channels