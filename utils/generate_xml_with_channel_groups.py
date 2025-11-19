import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty = reparsed.toprettyxml(indent=" ", encoding='utf-8').decode('utf-8')
    # Remove extra blank lines that minidom adds
    lines = [line for line in pretty.split('\n') if line.strip()]
    return '\n'.join(lines) + '\n'

def generate_xml_with_channel_groups(template_xml_path, output_xml_path, channel_groups,  
                                      date=None, group_regions=None):
    """Generate a new XML file with channel groups identified by get_channel_groups function.
    
    Args:
        template_xml_path: Path to the template XML file
        output_xml_path: Path to save the new XML file
        channel_groups: List of lists, where each inner list contains indices belonging to the same group
        date: Optional date string to set in the XML (format: YYYY-MM-DD)
        group_regions: Optional list of brain region names, one per channel group. 
                       Must match the number of detected groups. E.g. ['CA1', 'CA3', 'PFC', 'TH', 'ACC']
    """
    
    print(f"\nGenerating XML with {len(channel_groups)} channel groups")
    for idx, group in enumerate(channel_groups):
        print(f"  Group {idx}: {len(group)} channels")
    
    # Validate group_regions if provided
    if group_regions is not None:
        if len(group_regions) != len(channel_groups):
            raise ValueError(
                f"Number of group_regions ({len(group_regions)}) must match "
                f"number of detected channel groups ({len(channel_groups)})"
            )
    
    # Parse the template XML
    tree = ET.parse(template_xml_path)
    root = tree.getroot()
    
    # Find the anatomicalDescription section
    anat_desc = root.find('anatomicalDescription')
    if anat_desc is None:
        raise ValueError("No anatomicalDescription section found in template XML")
    
    # Remove existing channelGroups
    channel_groups_elem = anat_desc.find('channelGroups')
    if channel_groups_elem is not None:
        anat_desc.remove(channel_groups_elem)
    
    # Create new channelGroups element
    channel_groups_elem = ET.SubElement(anat_desc, 'channelGroups')
    
    # Add each group
    for group_channels in channel_groups:
        group_elem = ET.SubElement(channel_groups_elem, 'group')
        
        for channel_idx in group_channels:
            channel_elem = ET.SubElement(group_elem, 'channel')
            channel_elem.set('skip', '0')
            # get_channel_groups returns 0-indexed channel indices
            # XML also uses 0-indexed channel numbers, so use directly
            channel_elem.text = str(channel_idx)
    
    # Set date if provided
    if date is not None:
        general_info = root.find('generalInfo')
        if general_info is not None:
            date_elem = general_info.find('date')
            if date_elem is not None:
                date_elem.text = str(date)
                print(f"Set date to: {date}")
            else:
                print("Warning: date element not found in generalInfo section")
        else:
            print("Warning: generalInfo section not found in template")
    
    # Build brain regions from group_regions if provided
    # Generate BOTH per-channel region array AND brainRegions structure
    if group_regions is not None and len(group_regions) > 0:
        # Get total number of channels
        acq_system = root.find('acquisitionSystem')
        if acq_system is not None:
            n_channels_elem = acq_system.find('nChannels')
            if n_channels_elem is not None:
                n_channels = int(n_channels_elem.text)
                
                # Create a dict mapping region names to {channels: [], electrodeGroups: []}
                from collections import defaultdict
                regions_dict = defaultdict(lambda: {'channels': [], 'electrodeGroups': []})
                
                # Create per-channel region array for sessionInfo.region
                region_array = [''] * n_channels
                
                # Assign channels and electrode groups based on channel groups
                for group_idx, region_name in enumerate(group_regions):
                    if region_name:  # Skip empty region names
                        regions_dict[region_name]['channels'].extend(channel_groups[group_idx])
                        regions_dict[region_name]['electrodeGroups'].append(group_idx)
                        
                        # Also populate per-channel region array
                        for ch in channel_groups[group_idx]:
                            region_array[ch] = region_name
                
                # 1. Set per-channel region array (for sessionInfo.region - MATLAB code expects this)
                region_elem = root.find('region')
                if region_elem is not None:
                    region_elem.text = ' '.join(region_array)
                    print(f"Set per-channel region array ({n_channels} channels)")
                
                # 2. Set brainRegions structure (for NeuroScope2 - expects .channels and .electrodeGroups)
                brain_regions_elem = root.find('brainRegions')
                if brain_regions_elem is not None:
                    # Clear existing regions
                    brain_regions_elem.clear()
                    
                    # Create XML structure for each brain region
                    for region_name, data in regions_dict.items():
                        region_node = ET.SubElement(brain_regions_elem, region_name)
                        
                        # Add channels element
                        channels_elem = ET.SubElement(region_node, 'channels')
                        channels_elem.text = ' '.join(str(ch) for ch in sorted(data['channels']))
                        
                        # Add electrodeGroups element
                        electrode_groups_elem = ET.SubElement(region_node, 'electrodeGroups')
                        electrode_groups_elem.text = ' '.join(str(eg) for eg in sorted(data['electrodeGroups']))
                        
                        print(f"Brain region '{region_name}': {len(data['channels'])} channels, electrode groups {data['electrodeGroups']}")
                else:
                    print("Warning: brainRegions section not found in template")
            else:
                print("Warning: nChannels not found in acquisitionSystem")
        else:
            print("Warning: acquisitionSystem section not found")
    
    # Write the new XML
    # Using prettify for better formatting
    pretty_xml = prettify_xml(root)
    
    # Write to file
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    print(f"\nGenerated XML file: {output_xml_path}")
    print(f"Total groups: {len(channel_groups)}")
    total_channels = sum(len(g) for g in channel_groups)
    print(f"Total channels: {total_channels}")

if __name__ == '__main__':
    # Example usage - update paths as needed
    
    # Generate XML for NPX3 channel groups
    # Uncomment and update the path to your channel_positions.npy file
    # generate_xml_with_channel_groups(
    #     template_xml_path='sample_xml_neuroscope.xml',
    #     output_xml_path='NPX3_neuroscope_with_channel_groups.xml',
    #     npy_path='/path/to/your/channel_positions.npy',
    #     x_threshold=50,
    #     y_threshold=50
    # )
    
    print("Usage examples:")
    print("\n# Example 1: Assign brain regions to channel groups")
    print("# If 5 channel groups are detected, provide a list of 5 region names")
    print("generate_xml_with_channel_groups(")
    print("    template_xml_path='sample_xml_neuroscope.xml',")
    print("    output_xml_path='output.xml',")
    print("    npy_path='/path/to/channel_positions.npy',")
    print("    date='2025-11-17',")
    print("    group_regions=['CA1', 'CA3', 'PFC', 'TH', 'ACC']  # One per group")
    print(")")
    print("\n# Example 2: Same region for multiple groups")
    print("# If groups 0,1,2 are all in CA1, and groups 3,4 are in PFC:")
    print("generate_xml_with_channel_groups(")
    print("    template_xml_path='sample_xml_neuroscope.xml',")
    print("    output_xml_path='output.xml',")
    print("    npy_path='/path/to/channel_positions.npy',")
    print("    group_regions=['CA1', 'CA1', 'CA1', 'PFC', 'PFC']")
    print(")")
    print("\n# This generates:")
    print("# <brainRegions>")
    print("#   <CA1>")
    print("#     <channels>0 1 2 3 4 5 6 7 8</channels>  (all channels from groups 0,1,2)")
    print("#     <electrodeGroups>0 1 2</electrodeGroups>")
    print("#   </CA1>")
    print("#   <PFC>")
    print("#     <channels>9 10 11 12</channels>  (all channels from groups 3,4)")
    print("#     <electrodeGroups>3 4</electrodeGroups>")
    print("#   </PFC>")
    print("# </brainRegions>")

