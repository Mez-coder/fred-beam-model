#!/usr/bin/env python3
"""Convert DICOM CT series to MHD, optionally cropped to a structure."""
import argparse
import os
import numpy as np
np.NaN = np.nan # thanks for nothing FredTools 
import fredtools as ft

import SimpleITK as sitk


# Default ROI types to include
DEFAULT_ROI_TYPES = ['ORGAN', 'CTV']

# Processing order priority (higher = processed later = takes precedence)
ROI_TYPE_PRIORITY = {
    'EXTERNAL': 0,
    'ORGAN': 1,
    'CTV': 2,
    'PTV': 3,
    'GTV': 4,
}


def print_structure_table(rs_info, included_ids=None, title="Available Structures"):
    """Print the structure table with optional highlighting of included structures."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)
    
    if included_ids is not None:
        print(f"{'ID':<6}{'ROIType':<15}{'ROIName':<25}{'Included':<10}")
        print('-'*80)
        for roi_id, row in rs_info.iterrows():
            included = "YES" if roi_id in included_ids else ""
            print(f"{roi_id:<6}{row['ROIType']:<15}{row['ROIName']:<25}{included:<10}")
    else:
        print(f"{'ID':<6}{'ROIType':<15}{'ROIName':<25}")
        print('-'*80)
        for roi_id, row in rs_info.iterrows():
            print(f"{roi_id:<6}{row['ROIType']:<15}{row['ROIName']:<25}")
    print('='*80 + '\n')


def parse_id_input(input_str, valid_ids):
    """Parse comma-separated ID input and validate against valid IDs."""
    if not input_str.strip():
        return []
    
    ids = []
    for part in input_str.split(','):
        part = part.strip()
        if part:
            try:
                id_val = int(part)
                if id_val in valid_ids:
                    ids.append(id_val)
                else:
                    print(f"  Warning: ID {id_val} not found, skipping")
            except ValueError:
                print(f"  Warning: '{part}' is not a valid ID, skipping")
    return ids


def get_structure_selection_interactive(rs_info):
    """Interactive workflow to select which structures to include."""
    all_ids = set(rs_info.index.tolist())
    
    print_structure_table(rs_info, title="All Available Structures")
    
    # Default inclusion based on ROI type
    default_included_ids = set()
    for roi_id, row in rs_info.iterrows():
        if row['ROIType'].upper() in [t.upper() for t in DEFAULT_ROI_TYPES]:
            default_included_ids.add(roi_id)
    
    print(f"Default ROI types included: {DEFAULT_ROI_TYPES}")
    print_structure_table(rs_info, included_ids=default_included_ids, title="Default Included Structures")
    
    # ADD structures
    print("STEP 1: ADD additional structures")
    print("Enter structure IDs to ADD (comma-separated), or press Enter to skip:")
    add_input = input("  Add IDs: ").strip()
    add_ids = parse_id_input(add_input, all_ids)
    if add_ids:
        print(f"  Adding structures: {add_ids}")
    
    included_ids = default_included_ids.union(set(add_ids))
    
    # REMOVE structures
    print_structure_table(rs_info, included_ids=included_ids, title="Current Included Structures")
    print("STEP 2: REMOVE structures from selection")
    print("Enter structure IDs to REMOVE (comma-separated), or press Enter to skip:")
    remove_input = input("  Remove IDs: ").strip()
    remove_ids = parse_id_input(remove_input, included_ids)
    if remove_ids:
        print(f"  Removing structures: {remove_ids}")
    
    included_ids = included_ids - set(remove_ids)
    
    # BACKGROUND structures
    print_structure_table(rs_info, included_ids=included_ids, title="Final Included Structures")
    print("STEP 3: Select BACKGROUND structures (processed first, will be overwritten by others)")
    print("Enter structure IDs for background (comma-separated), or press Enter to skip:")
    bg_input = input("  Background IDs: ").strip()
    background_ids = set(parse_id_input(bg_input, included_ids))
    if background_ids:
        print(f"  Background structures (processed first): {background_ids}")
    
    return included_ids, background_ids


def get_structure_selection_automated(rs_info, include_ids=None, exclude_ids=None, background_ids=None):
    """Automated workflow to select structures based on command-line arguments."""
    all_ids = set(rs_info.index.tolist())
    
    # Start with default inclusion based on ROI type
    included_ids = set()
    for roi_id, row in rs_info.iterrows():
        if row['ROIType'].upper() in [t.upper() for t in DEFAULT_ROI_TYPES]:
            included_ids.add(roi_id)
    
    # Add specified IDs
    if include_ids:
        for id_val in include_ids:
            if id_val in all_ids:
                included_ids.add(id_val)
            else:
                print(f"  Warning: Include ID {id_val} not found, skipping")
    
    # Remove specified IDs
    if exclude_ids:
        for id_val in exclude_ids:
            if id_val in included_ids:
                included_ids.discard(id_val)
    
    # Validate background IDs
    bg_ids = set()
    if background_ids:
        for id_val in background_ids:
            if id_val in included_ids:
                bg_ids.add(id_val)
    
    return included_ids, bg_ids


def sort_structures_for_processing(rs_info, included_ids, background_ids):
    """Sort structures for processing order."""
    background_list = []
    regular_list = []
    
    for roi_id in included_ids:
        row = rs_info.loc[roi_id]
        roi_type = row['ROIType'].upper()
        priority = ROI_TYPE_PRIORITY.get(roi_type, 1)
        
        if roi_id in background_ids:
            background_list.append((roi_id, row, priority))
        else:
            regular_list.append((roi_id, row, priority))
    
    background_list.sort(key=lambda x: x[2])
    regular_list.sort(key=lambda x: x[2])
    
    processing_order = [(roi_id, row) for roi_id, row, _ in background_list]
    processing_order.extend([(roi_id, row) for roi_id, row, _ in regular_list])
    
    return processing_order


def create_structure_masks(ct_image, rs_file, rs_info, included_ids, background_ids):
    """Create a combined mask image containing selected structures."""
    processing_order = sort_structures_for_processing(rs_info, included_ids, background_ids)
    
    print(f"\nProcessing {len(processing_order)} structures in order:")
    for i, (roi_id, row) in enumerate(processing_order, 1):
        bg_marker = " [BACKGROUND]" if roi_id in background_ids else ""
        print(f"  {i}. {row['ROIName']} ({row['ROIType']}){bg_marker}")
    print()
    
    ref_array = sitk.GetArrayFromImage(ct_image)
    combined_array = np.zeros(ref_array.shape, dtype=np.uint16)
    
    lookup_dict = {}
    
    for idx, (roi_id, row) in enumerate(processing_order, start=1):
        structure_name = row['ROIName']
        roi_type = row['ROIType']
        is_background = roi_id in background_ids
        
        try:
            mask = ft.mapStructToImg(ct_image, rs_file, structure_name, CPUNo='none', displayInfo=False)
            mask_array = sitk.GetArrayFromImage(mask)
            combined_array[mask_array > 0] = idx
            lookup_dict[idx] = {
                'ROIName': structure_name,
                'ROIType': roi_type,
                'IsBackground': is_background
            }
            print(f"  Added structure {idx}: {structure_name} ({roi_type})")
        except Exception as e:
            print(f"  Warning: Could not process structure '{structure_name}': {e}")
            continue
    
    combined_mask = sitk.GetImageFromArray(combined_array)
    combined_mask.CopyInformation(ct_image)
    
    return combined_mask, lookup_dict


def save_lookup_table(lookup_dict, output_path):
    """Save the structure lookup table to a text file."""
    with open(output_path, 'w') as f:
        f.write("# Structure Lookup Table\n")
        f.write("# Integer Value\tROIName\tROIType\tIsBackground\n")
        f.write("# 0 = Background (no structure)\n")
        for int_val, info in sorted(lookup_dict.items()):
            bg_flag = "True" if info['IsBackground'] else "False"
            f.write(f"{int_val}\t{info['ROIName']}\t{info['ROIType']}\t{bg_flag}\n")
    print(f"Saved lookup table: {output_path}")



def extract_dose(rd_files, rn_file, number_fractions):
    """Extract RTDOSE per field and return list of (dose_image, field_info) tuples.
    
    Each dose is normalized per fraction and corrected for RBE (divided by 1.1).
    Field info includes gantry angle, couch angle, and field name for naming.
    """
    if not rd_files:
        print("No RTDOSE files found")
        return None

    # Get field information from the RN plan
    rn_fields = ft.getRNFields(rn_file, displayInfo=True)
    print(f"\nFound {len(rn_fields)} fields in RN plan:")
    print(rn_fields.to_string())
    print()

    dose_results = []

    for _, field in rn_fields.iterrows():
        field_number = field['FNo']
        field_name = field.get('FName', f'Field{field_number}')
        gantry_angle = field.get('FGantryAngle', 0.0)
        couch_angle = field.get('FCouchAngle', 0.0)

        try:
            rd_file = ft.getRDFileNameForFieldNumber(rd_files, field_number, displayInfo=False)
        except Exception as e:
            print(f"  Warning: No RD file for field {field_number} ({field_name}): {e}")
            continue

        print(f"  Loading dose for field {field_number}: {field_name} "
              f"(Gantry={gantry_angle:.1f}°, Couch={couch_angle:.1f}°)")

        dose_img = ft.getRD(rd_file, displayInfo=False)
        # Optionally Convert from Absolute Dose to dose per fraction and remove RBE factor (1.1) so we are saving D2W TOTAL
        dose_img = sitk.Cast(dose_img, sitk.sitkFloat32) #  / number_fractions / 1.1

        field_info = {
            'fieldNumber': field_number,
            'fieldName': field_name,
            'gantryAngle': gantry_angle,
            'couchAngle': couch_angle,
        }
        dose_results.append((dose_img, field_info))

    print(f"\nSuccessfully loaded {len(dose_results)} field doses")
    return dose_results

def resample_to_reference(image, reference, interpolator=sitk.sitkLinear, default_value=0.0):
    """Resample image to match reference image geometry."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(image)


def main():
    parser = argparse.ArgumentParser(description="Convert DICOM CT to MHD (HFS patients only)")
    parser.add_argument("--dcm_folder", "-d", default="dcm", help="Path to DICOM folder")
    parser.add_argument("--structure", "-s", help="Structure name to crop to")
    parser.add_argument("--spacing", "-r", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Resample to spacing in mm (e.g., 2 2 2)")
    parser.add_argument("--output", "-o", default="ct.mhd", help="Output CT filename")
    
    # Structure mask arguments
    parser.add_argument("--create_structures", "-cs", action='store_true',
                        help="Create structure mask MHD file")
    parser.add_argument("--output_structures", "-os", default="structures.mhd", 
                        help="Output structures filename")
    parser.add_argument("--output_lookup", "-ol", default="structure_lookup.txt",
                        help="Output lookup table filename")
    
    # Automated mode arguments
    parser.add_argument("--include_ids", "-ii", type=str, default=None,
                        help="Comma-separated IDs to include")
    parser.add_argument("--exclude_ids", "-ei", type=str, default=None,
                        help="Comma-separated IDs to exclude")
    parser.add_argument("--background_ids", "-bi", type=str, default=None,
                        help="Comma-separated IDs for background structures")
    
    # Dose extraction argument
    parser.add_argument("--get_dicom_dose", "-gd", action='store_true',
                        help="Extract RTDOSE and save as MHD")
    
    args = parser.parse_args()
    
    # Sort and load DICOM files
    dcm_file_dict = ft.sortDicoms(args.dcm_folder, recursive=False, displayInfo=True)   
    ct_files = dcm_file_dict['CTfileNames']
    rd_files = dcm_file_dict['RDfileNames']
    rs_file = dcm_file_dict['RSfileNames']
    rn_file = dcm_file_dict['RNfileNames']
    
    # Load CT
    ct = ft.getCT(ct_files, displayInfo=True)
    
    # Get fraction info
    rn_info = ft.getRNInfo(rn_file)
    num_frac = rn_info["fractionNo"]

    print("number of fractions: ", num_frac)
    
    
    # Extract dose if requested
    dose = None
    # Extract dose if requested
    if args.get_dicom_dose:
        print("\nExtracting RTDOSE...")
        dose_results = extract_dose(rd_files, rn_file, num_frac)
        if dose_results is not None:
            output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
            for dose_img, field_info in dose_results:
                # Resample to CT grid
                dose_img = resample_to_reference(dose_img, ct)
                
                # Crop if structure specified
                if args.structure:
                    mask = ft.mapStructToImg(ct, rs_file, args.structure, CPUNo='none', displayInfo=False)
                    dose_img = ft.cropImgToMask(dose_img, mask, displayInfo=False)
                
                # Resample if spacing specified
                if args.spacing:
                    dose_img = ft.resampleImg(dose_img, args.spacing, interpolation="linear", displayInfo=False)
                
                # Build filename: e.g. "dose_G90.0_C0.0_FieldName.mhd"
                g = field_info['gantryAngle']
                c = field_info['couchAngle']
                name = field_info['fieldName'].replace(' ', '_')
                dose_path = os.path.join(output_dir, "dose_to_water_total", f"{name}.mhd")

                os.makedirs(os.path.dirname(dose_path), exist_ok=True)

                ft.writeMHD(dose_img, dose_path, displayInfo=False)
                print(f"Saved dose: {dose_path}")
    
    # Create structure masks if requested
    structures = None
    lookup_dict = None
    
    if args.create_structures:
        rs_info = ft.getRSInfo(rs_file, displayInfo=False)


        
        automated_mode = any([args.include_ids, args.exclude_ids, args.background_ids])
        
        if automated_mode:
            include_ids = [int(x.strip()) for x in args.include_ids.split(',')] if args.include_ids else None
            exclude_ids = [int(x.strip()) for x in args.exclude_ids.split(',')] if args.exclude_ids else None
            background_ids = [int(x.strip()) for x in args.background_ids.split(',')] if args.background_ids else None
            
            included_ids, bg_ids = get_structure_selection_automated(
                rs_info, include_ids, exclude_ids, background_ids
            )
        else:
            included_ids, bg_ids = get_structure_selection_interactive(rs_info)
        
        if included_ids:
            print("\nCreating structure masks...")
            structures, lookup_dict = create_structure_masks(ct, rs_file, rs_info, included_ids, bg_ids)
    
    # Crop to structure if specified
    if args.structure:
        mask = ft.mapStructToImg(ct, rs_file, args.structure, CPUNo='none', displayInfo=False)
        ct = ft.cropImgToMask(ct, mask, displayInfo=False)
        if structures is not None:
            structures = ft.cropImgToMask(structures, mask, displayInfo=False)
        if dose is not None:
            dose = ft.cropImgToMask(dose, mask, displayInfo=False)
    
    # Resample if spacing specified
    if args.spacing:
        ct = ft.resampleImg(ct, args.spacing, interpolation="linear", displayInfo=False)
        if structures is not None:
            structures = ft.resampleImg(structures, args.spacing, interpolation="nearest", displayInfo=False)
        if dose is not None:
            dose = ft.resampleImg(dose, args.spacing, interpolation="linear", displayInfo=False)
    
    # Save outputs
    ft.writeMHD(ct, args.output, displayInfo=False)
    print(f"Saved CT: {args.output}")
    
    if structures is not None and lookup_dict is not None:
        ft.writeMHD(structures, args.output_structures, displayInfo=False)
        print(f"Saved structures: {args.output_structures}")
        save_lookup_table(lookup_dict, args.output_lookup)
    
    
if __name__ == "__main__":
    main()