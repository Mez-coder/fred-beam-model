#!/usr/bin/env python3
"""
Generate FRED simulation input file (.inp) from DICOM RN treatment plan.

This script reads a DICOM RN (Radiotherapy Plan) file and a beam model file,
then generates a formatted .inp file for FRED Monte Carlo simulation.

Usage:
    python get_plan.py <dicom_folder> <beam_model_file> [output_file]

Example:
    python get_plan.py dicom_folder CustomBeamModel.bm plan.inp

Author: Generated for proton therapy workflow
"""

import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import pydicom
import fredtools as ft
import re


# =============================================================================
# Configuration Constants
# =============================================================================

# Nozzle geometry (in mm, will be converted to cm for output)
NOZZLE_TO_ISO_DISTANCE_MM = 600.0  # Nozzle exit to isocenter
SMX_TO_ISO_DISTANCE_MM = 2000.0    # SMX magnet to isocenter
SMY_TO_ISO_DISTANCE_MM = 2560.0    # SMY magnet to isocenter

# Convert to cm for .inp file
NOZZLE_TO_ISO_DISTANCE_CM = NOZZLE_TO_ISO_DISTANCE_MM / 10.0  # 60 cm

# Reference plane distance for field origin (SMY magnet to iso)
FIELD_ORIGIN_DISTANCE_CM = SMY_TO_ISO_DISTANCE_MM / 10.0  # 256 cm

# emittanceRefPlaneDistance - distance from isocenter to reference plane
EMITTANCE_REF_PLANE_DISTANCE_CM = (SMY_TO_ISO_DISTANCE_MM / 10 - NOZZLE_TO_ISO_DISTANCE_CM) #  194 cm


# =============================================================================
# Helper Functions
# =============================================================================

def get_table_top_positions(rn_file):
    """
    Extract table top positions from DICOM RN file using pydicom.
    
    The table top positions are stored in IonBeamSequence[n].IonControlPointSequence[0].
    We take the values from the first beam's first control point (they should be the same
    for all beams in a typical plan).
    
    Parameters
    ----------
    rn_file : str
        Path to DICOM RN file
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'vertical': TableTopVerticalPosition (mm)
        - 'longitudinal': TableTopLongitudinalPosition (mm)
        - 'lateral': TableTopLateralPosition (mm)
        Returns None values if not found.
    """
    ds = pydicom.dcmread(rn_file)
    
    positions = {
        'vertical': None,
        'longitudinal': None,
        'lateral': None
    }
    
    try:
        if "IonBeamSequence" in ds and len(ds.IonBeamSequence) > 0:
            beam = ds.IonBeamSequence[0]
            if "IonControlPointSequence" in beam and len(beam.IonControlPointSequence) > 0:
                cp = beam.IonControlPointSequence[0]
                
                if "TableTopVerticalPosition" in cp:
                    positions['vertical'] = float(cp.TableTopVerticalPosition)
                if "TableTopLongitudinalPosition" in cp:
                    positions['longitudinal'] = float(cp.TableTopLongitudinalPosition)
                if "TableTopLateralPosition" in cp:
                    positions['lateral'] = float(cp.TableTopLateralPosition)
    except Exception as e:
        print(f"Warning: Could not extract table top positions: {e}")
    
    return positions


def calculate_adjusted_isocenter(isocenter_mm, table_positions):
    """
    Calculate the adjusted isocenter by adding table top positions.
    
    The DICOM isocenter and table top positions are combined to get the 
    final phantom position for FRED.
    
    Coordinate mapping (DICOM to FRED):
    - DICOM X (Lateral) -> FRED X
    - DICOM Y (Longitudinal) -> FRED Y  
    - DICOM Z (Vertical) -> FRED Z
    
    Parameters
    ----------
    isocenter_mm : list
        Isocenter position [X, Y, Z] in mm from DICOM
    table_positions : dict
        Table top positions from get_table_top_positions()
    
    Returns
    -------
    list
        Adjusted isocenter [X, Y, Z] in mm
    """
    # Start with isocenter
    adjusted = list(isocenter_mm)
    
    # Add table top positions
    # TableTopLateralPosition -> X
    if table_positions['lateral'] is not None:
        adjusted[0] += table_positions['lateral']
    
    # TableTopLongitudinalPosition -> Y
    if table_positions['longitudinal'] is not None:
        adjusted[1] += table_positions['longitudinal']
    
    # TableTopVerticalPosition -> Z
    if table_positions['vertical'] is not None:
        adjusted[2] += table_positions['vertical']
    
    return adjusted


def interpolate_beam_params(bm_energy_df, energy):
    """
    Interpolate beam model parameters for a given energy.
    
    Parameters
    ----------
    bm_energy_df : pd.DataFrame
        Beam model energy DataFrame (indexed by nomEnergy)
    energy : float
        Nominal energy to interpolate for (MeV)
    
    Returns
    -------
    dict
        Dictionary with interpolated beam parameters
    """
    # Get the energy values from the index
    energies = bm_energy_df.index.values
    
    # Check bounds
    if energy < energies.min() or energy > energies.max():
        raise ValueError(f"Energy {energy} MeV is outside beam model range [{energies.min()}, {energies.max()}] MeV")
    
    # Interpolate each parameter
    params = {}
    for col in bm_energy_df.columns:
        params[col] = np.interp(energy, energies, bm_energy_df[col].values)
    
    return params


def calculate_field_vectors(gantry_angle_deg):
    """
    Calculate field direction vectors (f, u) from gantry angle.
    
    For a gantry rotating around the Z-axis (IEC convention):
    - Gantry at 0°: beam comes from +Y direction (anterior)
    - Gantry at 90°: beam comes from -X direction (left lateral)
    - Gantry at 180°: beam comes from -Y direction (posterior)
    - Gantry at 270°: beam comes from +X direction (right lateral)
    
    Parameters
    ----------
    gantry_angle_deg : float
        Gantry angle in degrees
    
    Returns
    -------
    tuple
        (f_vector, u_vector) - forward and up direction vectors
    """
    # For now, use fixed vectors as in the example
    # f points in beam direction, u points up
    # TODO: Calculate properly based on gantry angle if needed
    f = [0.0, 1.0, 0.0]
    u = [0.0, 0.0, 1.0]
    
    return f, u


def format_float(value, width=12, precision=4, sign=True):
    """Format a float with consistent width and sign."""
    if sign:
        return f"{value:+{width}.{precision}f}"
    else:
        return f"{value:{width}.{precision}f}"


def format_scientific(value, precision=10):
    """Format a float in scientific notation."""
    return f"{value:+.{precision}E}"

import shutil

def clear_directories(*dirs):
    """Clear all files in directories without removing the directories themselves."""
    for d in dirs:
        path = Path(d)
        if path.exists():
            for file in path.iterdir():
                if file.is_file():
                    file.unlink()
        else:
            path.mkdir(parents=True)

# =============================================================================
# Main INP Generation Functions
# =============================================================================

def generate_header(beam_model, fields_info, spots_df, total_primaries_per_field):
    """Generate the .inp file header section."""
    lines = []
    
    # Get beam model description
    bm_desc = beam_model.get('BM Description', {})
    bm_name = bm_desc.get('name', 'Unknown')
    bm_time = bm_desc.get('creationTime', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    
    # Count fields and spots
    num_fields = len(fields_info)
    spots_per_field = [len(spots_df[spots_df['FDeliveryNo'] == fid]) for fid in fields_info['FDeliveryNo']]
    total_spots = sum(spots_per_field)
    total_primaries = sum(total_primaries_per_field)
    
    # Format header
    lines.append("#" * 143)
    lines.append(f"# Beam model: {bm_name} ({bm_time})")
    lines.append(f"# Fields no.: {num_fields}")
    lines.append(f"# Fields PB no.: {spots_per_field} (total: {total_spots})")
    
    # Format primaries
    prim_list = [f"{p:.2f}" for p in total_primaries_per_field]
    lines.append(f"# Fields prim. no.: [{', '.join(prim_list)}] (total: {total_primaries:.2f} ~= {total_primaries:.3E})")
    lines.append("#" * 143)
    
    return lines


def generate_field_definitions(fields_info):
    """Generate field definition lines."""
    lines = []
    
    for _, field in fields_info.iterrows():
        field_id = int(field['FDeliveryNo'])
        field_name = field['FName']
        
        # Field origin - at SMY magnet position (256 cm from iso)
        # O is defined in the FRED coordinate system
        O = [0.0, -FIELD_ORIGIN_DISTANCE_CM, 0.0]
        
        # Get direction vectors
        f, u = calculate_field_vectors(field['FGantryAngle'])
        
        # Format the line
        O_str = f"[ {format_float(O[0])}, {format_float(O[1])}, {format_float(O[2])} ]"
        f_str = f"[ {format_float(f[0], precision=3)}, {format_float(f[1], precision=3)}, {format_float(f[2], precision=3)} ]"
        u_str = f"[ {format_float(u[0], precision=3)}, {format_float(u[1], precision=3)}, {format_float(u[2], precision=3)} ]"
        
        line = f"field: {field_id} ; O={O_str} ;   f={f_str} ;   u={u_str} ;   # Definition of field {field_id} (field name: '{field_name}')"
        lines.append(line)
    
    return lines


def generate_pbmaster_definitions(fields_info):
    """Generate pbmaster definition lines."""
    lines = []
    
    for _, field in fields_info.iterrows():
        field_id = int(field['FDeliveryNo'])
        
        line = (f"pbmaster: {field_id} ; particle=proton ; Xsec=emittance ; "
                f"emittanceRefPlaneDistance={EMITTANCE_REF_PLANE_DISTANCE_CM:.4f} ; "
                f"columns=[P.x, P.y, P.z, v.x, v.y, v.z, Emean, Estdev, N, "
                f"twissAlphaX, twissBetaX, emittanceX, twissAlphaY, twissBetaY, emittanceY]")
        lines.append(line)
    
    return lines


def generate_group_region_definitions(fields_info):
    """Generate group and region definitions."""
    lines = []
    
    # Create field group string
    field_names = " ".join([f"field_{int(f['FDeliveryNo'])}" for _, f in fields_info.iterrows()])
    
    lines.append(f"group: fieldGroup {field_names}")
    lines.append("region: gantry ; O = [0,0,0] ; L = [1,1,1] ; lAdaptiveSize=t ; material = vacuum")
    lines.append("set_parent: gantry fieldGroup nozzleGroup")
    lines.append("save_regions: 0")
    
    return lines


def clean_rs_id(rs_id):
    """
    Clean range shifter ID to match region naming convention.
    
    Converts DICOM RS ID (e.g., "RS=2cm") to region name (e.g., "RS_2cm")
    """
    if rs_id is None or pd.isna(rs_id):
        return None
    
    # Convert "RS=2cm" -> "RS_2cm"
    cleaned = str(rs_id).replace("=", "_") # TODO fix
    cleaned = "RS"
    return cleaned

def extract_rs_value(rs_id):
    """
    Extract the numeric value from a range shifter ID.
    
    e.g., "RS=2cm" -> 2
    """
    if rs_id is None or pd.isna(rs_id):
        return None
    
    match = re.search(r'\d+', str(rs_id))
    return int(match.group()) if match else None

def generate_setup_delivery_sequence(fields_info, isocenter_mm):
    """Generate setup and delivery sequence for all fields."""
    lines = []
    rs_ids = []
    
    lines.append("#" * 50)
    lines.append("###### Start of setup and delivery sequence ######")
    lines.append("#" * 50)
    
    # Convert isocenter from mm to cm AND flip the signs
    # DICOM isocenter [40, -240, -570] mm -> [-4, 24, 57] cm in FRED
    iso_cm = [-x / 10.0 for x in isocenter_mm]
    
    for _, field in fields_info.iterrows():
        field_id = int(field['FDeliveryNo'])
        field_name = field['FName']
        gantry_angle = field['FGantryAngle']
        couch_angle = field['FCouchAngle']
        snout_pos_mm = field['PBSnoutPos']
        rs_id = field.get('PBRSID', None)
        rs_setting = field.get('PBRSSetting', None)
        
        # Convert snout position to cm
        snout_pos_cm = snout_pos_mm / 10.0
        
        # Calculate snout shift (snout position relative to nozzle)
        # The snout moves along the beam direction (Y in FRED coords)
        snout_shift_cm = NOZZLE_TO_ISO_DISTANCE_CM - snout_pos_cm
        
        lines.append(f"###### setup sequence for field {field_id} (field name: '{field_name}')")
        
        # Nozzle translation
        lines.append("# translate nozzle regions by snout position")
        lines.append(f"transform: nozzleGroup shift_by {format_float(0.0)} {format_float(-snout_shift_cm)} {format_float(0.0)}")
        
        # Gantry rotation
        lines.append("# rotate gantry by gantry angle")
        lines.append(f"transform: gantry rotate z {format_float(gantry_angle, sign=False)}")
        
        # Phantom translation and rotation
        lines.append("# translate and rotate phantom by iso position and couch rotation")
        lines.append(f"transform: phantom shift_by {format_float(iso_cm[0])} {format_float(iso_cm[1])} {format_float(iso_cm[2])}")
        lines.append(f"transform: phantom rotate y {format_float(-couch_angle)}")
        
        # Delivery sequence
        lines.append(f"# delivery sequences for slices in field {field_id}")
        lines.append("deactivate: fieldGroup, nozzleGroup")
        
        rs_ids.append(rs_id)
        
        # Activate range shifter if present and IN
        rs_name = clean_rs_id(rs_id)
        if rs_name and rs_setting == 'IN':
            lines.append(f"activate: field_{field_id} {rs_name}")
        else:
            lines.append(f"activate: field_{field_id}")
        
        lines.append(f"deliver: field_{field_id}")
        
        # Reset
        lines.append("# reset fieldGroup, nozzleGroup and phantom to default FoR")
        lines.append("restore: 0")
    
    lines.append("#" * 50)
    lines.append("###### End of setup and delivery sequence ########")
    lines.append("#" * 50)
    
    return lines, rs_ids



def generate_pb_definitions(spots_df, field_name_required, bm_energy_df, rn_plan):
    """Generate pencil beam definition lines."""
    lines = []
    primaries_per_field = {}  # Track N totals by field_id
    lines.append("#" * 36)
    lines.append("###### Start of PB definition ######")
    lines.append("#" * 36)
    
    # Group spots by field
    fields = spots_df['FDeliveryNo'].unique()
    
    spot_counter = 0
    
    for field_id in sorted(fields):
        
        
        field_spots = spots_df[spots_df['FDeliveryNo'] == field_id].copy()
        field_name = field_spots.iloc[0]['FName']

        if field_name != field_name_required:
            continue
        
        # Filter out spots with zero meterset weight
        field_spots = field_spots[field_spots['PBMsW'] > 0]
        
        if len(field_spots) == 0:
            continue
        
        primaries_per_field[field_id] = 0.0
        
        


        # Get BeamMeterset for this field from FractionGroupSequence
        beam_meterset = None
        for rbs in rn_plan.FractionGroupSequence[0].ReferencedBeamSequence:
            if rbs.ReferencedBeamNumber == field_id:
                beam_meterset = float(rbs.BeamMeterset)
                break
        
        # Get FinalCumulativeMetersetWeight from IonBeamSequence
        final_cumulative_meterset_weight = None
        for beam in rn_plan.IonBeamSequence:
            if beam.BeamNumber == field_id:
                final_cumulative_meterset_weight = float(beam.FinalCumulativeMetersetWeight)
                break
        
        if beam_meterset is None or final_cumulative_meterset_weight is None:
            print(f"Warning: Could not find meterset info for field {field_id}")
            continue


        
        lines.append(f"###### PB definitions for field {field_id}")
        lines.append("# spotID    fieldID           P.x           P.y           P.z             v.x           v.y           v.z        Emean     Estdev           N               twissAlphaX       twissBetaX        emittanceX          twissAlphaY       twissBetaY        emittanceY")
        
        for _, spot in field_spots.iterrows():
            spot_counter += 1
            
            # Get spot position (convert mm to cm)
            # DICOM PBPosX/PBPosY are in the plane perpendicular to beam
            # These map to P.x and P.y in FRED, P.z = 0 at reference plane
            px = -spot['PBPosX'] / 10.0  # mm to cm                           -------lmfao wtf?
            py = +spot['PBPosY'] / 10.0  # mm to cm
            pz = 0.0  # At the reference plane
            
            # Direction vector (simplified - pointing along beam)
            vx, vy, vz = 0.0, 0.0, 1.0
            
            # Energy
            energy = spot['PBnomEnergy']
            
            # Interpolate beam model parameters for this energy
            try:
                bm_params = interpolate_beam_params(bm_energy_df, energy)
            except ValueError as e:
                print(f"Warning: {e}. Skipping spot.")
                continue
            
            # Energy spread from beam model
            estdev = bm_params['dEnergy']
            
            # Calculate N (number of primaries)
            # N = scan_spot_meterset_weight * scaling_factor * (beam_meterset / final_cumulative_meterset_weight)
            msw = spot['PBMsW']
            scaling_factor = bm_params['scalingFactor']
            N = msw * scaling_factor * (beam_meterset / final_cumulative_meterset_weight)
            primaries_per_field[field_id] += N
            # Twiss parameters from beam model
            alpha_x = bm_params['alphaX']
            beta_x = bm_params['betaX']
            epsilon_x = bm_params['epsilonX']
            alpha_y = bm_params['alphaY']
            beta_y = bm_params['betaY']
            epsilon_y = bm_params['epsilonY']

            
            
            
            # Format the line
            line = (f"pb: {spot_counter:<10d} {int(field_id):<4d} \t"
                    f"{format_scientific(px)} {format_scientific(py)} {format_scientific(pz)}   "
                    f"{format_scientific(vx)} {format_scientific(vy)} {format_scientific(vz)}   "
                    f"{energy:7.3f} {estdev:.5E}   {N:.10E}   "
                    f"{format_scientific(alpha_x)} {format_scientific(beta_x)} {format_scientific(epsilon_x)}   "
                    f"{format_scientific(alpha_y)} {format_scientific(beta_y)} {format_scientific(epsilon_y)}")
            
            lines.append(line)
    
    # Convert to sorted list for header
    sorted_field_ids = sorted(primaries_per_field.keys())
    primaries_list = [primaries_per_field[fid] for fid in sorted_field_ids]
    
    return lines, primaries_list



def extract_field_info(spots_df):
    """Extract unique field information from spots DataFrame."""
    # Get unique fields based on FDeliveryNo
    fields_info = spots_df.groupby('FDeliveryNo').first().reset_index()
    
    # Keep only relevant columns
    cols_to_keep = ['FDeliveryNo', 'FNo', 'FName', 'FGantryAngle', 'FCouchAngle', 
                    'FCouchPitchAngle', 'FCouchRollAngle', 'FIsoPos', 'PBSnoutPos',
                    'PBRSID', 'PBRSSetting']
    
    # Filter to columns that exist
    cols_to_keep = [c for c in cols_to_keep if c in fields_info.columns]
    fields_info = fields_info[cols_to_keep]
    
    return fields_info

def generate_inp_file(rn_file, bm_file, output_dir="rtplans"):
    """
    Generate FRED .inp file(s) from DICOM RN and beam model files.
    Returns a list of output filenames (one per field).
    """
    print(f"Reading beam model: {bm_file}")
    beam_model = ft.readBeamModel(bm_file)
    bm_energy_df = beam_model['BM Energy']

    print(f"Reading RN DICOM: {rn_file}")

    rn_info = ft.getRNInfo(rn_file)
    spots_df = ft.getRNSpots(rn_file)
    isocenter = ft.getRNIsocenter(rn_file)

    table_positions = get_table_top_positions(rn_file)
    print(f"  Table top positions (mm) - for reference only:")
    print(f"    Vertical: {table_positions['vertical']}")
    print(f"    Longitudinal: {table_positions['longitudinal']}")
    print(f"    Lateral: {table_positions['lateral']}")
    print(f"  Isocenter (mm): {isocenter}")

    fields_info = extract_field_info(spots_df)
    print(fields_info)

    print(f"  Plan: {rn_info.get('planLabel', 'Unknown')}")
    print(f"  Patient: {rn_info.get('patientName', 'Unknown')}")
    print(f"  Fields: {len(fields_info)}")
    print(f"  Total spots: {len(spots_df)}")
    print(f"  Non-zero spots: {len(spots_df[spots_df['PBMsW'] > 0])}")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_files = []

    for iter, field in fields_info.iterrows():
        field_name = field.get("FName")

        rn_plan = pydicom.dcmread(rn_file)
        pb_lines, primaries_per_field = generate_pb_definitions(
            spots_df, field_name, bm_energy_df, rn_plan
        )

        print(f"Generating .inp file sections for field '{field_name}'...")
        all_lines = []
        all_lines.extend(generate_header(beam_model, fields_info, spots_df, primaries_per_field))
        all_lines.extend(generate_field_definitions(fields_info))
        all_lines.extend(generate_pbmaster_definitions(fields_info))
        all_lines.extend(generate_group_region_definitions(fields_info))
        lines, rs_ids = generate_setup_delivery_sequence(fields_info, isocenter)
        all_lines.extend(lines)
        all_lines.extend(pb_lines)

        output_file = f"rtplan_{field_name}.inp"
        full_path = output_path / output_file

        print(f"Writing to: {full_path}")
        with open(full_path, 'w') as f:
            f.write('\n'.join(all_lines))

        print(f"  Done: {output_file}")
        output_files.append(output_file)

    return output_files, rs_ids


def generate_fred_inp(rtplan_inp_filename, regions_filename, output_dir='freds', ct_file='CT.mhd',
                      nprim=1e4, ionization_potential=78.0,
                      materials_file='materials.inp', regions_file='regions.inp', rtplan_dir = "rtplans", regions_dir = "regions"):
    """
    Generate FRED main simulation input file for a single rtplan .inp.
    rtplan_inp_filename is just the filename (e.g. 'rtplan_Field1.inp'),
    not a full path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("### Phantom ###")
    lines.append("region<")
    lines.append("    ID=phantom")
    lines.append(f"    CTscan= '{ct_file}'")
    lines.append("    score = [dose, dose-to-water]")
    #lines.append("    scoreij = [dose]")
    lines.append("    lWriteLETd_parts = false")
    lines.append("    lWriteCTHU = true")
    lines.append("  ")
    lines.append("region>")
    lines.append("")

    lines.append("### Materials ###")
    lines.append(f"include: {materials_file}")
    lines.append("")

    lines.append("### Regions ###")
    lines.append(f"include: {regions_dir}/{regions_filename}")
    lines.append("")

    lines.append("### Beam ###")
    lines.append(f"include: {rtplan_dir}/{rtplan_inp_filename}")
    lines.append(f"nprim = {nprim:.0E}")
    lines.append("")

    lines.append("### Physics ###")
    lines.append(f"IonizPotential={int(ionization_potential)}")
    lines.append("")

    lines.append("### Control ###")
    lines.append("lAllowHUClamping = true")
    lines.append("lTracking_nuc_el = true")
    lines.append("lTracking_nuc_inel = true")
    lines.append("lTracking_nuc = true")
    lines.append("lTracking_fluc = true")
    lines.append("lUseInternalHU2Mat = true")
    lines.append(f"WaterIpot = {int(ionization_potential)}")
    lines.append("")

    # Derive fred filename from the rtplan filename
    # e.g. "rtplan_Field1.inp" -> "fred_Field1.inp"
    stem = rtplan_inp_filename
    if stem.startswith("rtplan_"):
        stem = stem[len("rtplan_"):]
    output_file = f"fred_{stem}"

    full_path = output_path / output_file

    with open(full_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  FRED input file written: {full_path}")
    return output_file



def generate_region_inp(rtplan_inp_filename, output_dir='regions', rs_length=None):
    """
    Generate FRED main simulation input file for a single rtplan .inp.
    rtplan_inp_filename is just the filename (e.g. 'rtplan_Field1.inp'),
    not a full path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not rs_length:
        rs_length = 1
        lines = []
        lines.append("region<")
        lines.append("  ID=RS")
        lines.append(f" L=[40, 40, {int(rs_length)}]")
        lines.append("  O=[0, 0, 0]")
        lines.append("  f=[0, 1, 0]")
        lines.append("  material=HU -1000")
        lines.append("  pivot=[0.5, 0.5, 1]")
        lines.append("  u=[0, 0, 1]")
        lines.append("  voxels=[1, 1, 1]")
        lines.append("region>")
        lines.append("")
        lines.append("group: rangeShifterGroup RS")
        lines.append("group: nozzleGroup rangeShifterGroup")
        lines.append("")

        
        
        
    else:
        lines = []
        lines.append("region<")
        lines.append("  ID=RS")
        lines.append(f" L=[40, 40, {int(rs_length)}]")
        lines.append("  O=[0, 0, 0]")
        lines.append("  f=[0, 1, 0]")
        lines.append("  material=uclhRS")
        lines.append("  pivot=[0.5, 0.5, 1]")
        lines.append("  u=[0, 0, 1]")
        lines.append("  voxels=[1, 1, 1]")
        lines.append("region>")
        lines.append("")
        lines.append("group: rangeShifterGroup RS")
        lines.append("group: nozzleGroup rangeShifterGroup")
        lines.append("")

    # Derive fred filename from the rtplan filename
    # e.g. "rtplan_Field1.inp" -> "fred_Field1.inp"
    stem = rtplan_inp_filename
    if stem.startswith("rtplan_"):
        stem = stem[len("rtplan_"):]
    output_file = f"region_{stem}"

    full_path = output_path / output_file

    with open(full_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  REGION input file written: {full_path}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate FRED .inp file from DICOM RN treatment plan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python get_plan.py dicom_folder CustomBeamModel.bm
        """
    )

    parser.add_argument('dcm_file', help='Path to DICOM folder')
    parser.add_argument('bm_file', help='Path to beam model (.bm) file')

    args = parser.parse_args()
    dcm_folder = args.dcm_file
    dcm_file_dict = ft.sortDicoms(dcm_folder, recursive=False, displayInfo=True)

    rn_file = dcm_file_dict['RNfileNames']

    if not Path(rn_file).exists():
        print(f"Error: RN file not found in: {args.dcm_file}")
        sys.exit(1)

    if not Path(args.bm_file).exists():
        print(f"Error: Beam model file not found: {args.bm_file}")
        sys.exit(1)

    try:
        clear_directories("rtplans", "freds", "regions")
        # Generate one rtplan .inp per field
        rtplan_files, rs_ids = generate_inp_file(rn_file, args.bm_file, output_dir="rtplans")

        # Generate one fred .inp per rtplan
        fred_files = []
        region_files = []
        a=0
        for rtplan_file in rtplan_files:
            rs_length = extract_rs_value(rs_ids[a])

            region_file = generate_region_inp(rtplan_file, output_dir="regions", rs_length=rs_length)
            fred_file = generate_fred_inp(rtplan_file, region_file, output_dir="freds", nprim=5e4)
            

            fred_files.append(fred_file)
            region_files.append(region_file)
            a+=1

        print(f"\nSuccessfully generated {len(rtplan_files)} rtplan file(s) in: rtplans/")
        for f in rtplan_files:
            print(f"  - {f}")

        print(f"\nSuccessfully generated {len(fred_files)} fred file(s) in: freds/")
        for f in fred_files:
            print(f"  - {f}")
        print(f"\nSuccessfully generated {len(region_files)} region file(s) in: regions/")
        for f in region_files:
            print(f"  - {f}")

    except Exception as e:
        print(f"Error generating .inp file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()