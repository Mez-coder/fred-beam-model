

import SimpleITK as sitk
import fredtools as ft
import matplotlib.pyplot as plt
import numpy as np

# ============= USER INPUTS =============

fields = ["Field1"]
sim1_dose = "FRED"

if sim1_dose == "FRED":

    sim1_ct = "CT.mhd"
    sim1_doses = [f"output/out_{field}/score/Phantom.DoseToWater.mhd" for field in fields]  # list format for consistency
else:
    sim1_ct = "CT.mhd"
    sim1_doses = [f"{field}.mhd" for field in fields]

sim2_ct = "dcm_data/output/ct.mhd"
sim2_doses = [f"dcm_data/output/{field}_AbsoluteDoseToWater.mhd" for field in fields] # AbsoluteDose.mhd" merged-Dose.mhd"

# the dose from Gate, merged-Dose is the dose as calc from Gate
# to get the treated dose, you multiply the gate dose by 2000 (hardcoded) * 1.1 (RBE) * n_frac to get the AbsoluteDose.mhd



dose1_scale_factor = 1.0 * 1.1 * 28 # * 1.1 * 30 # Scale dose 1 by 1.1 for RBE and n_frac to get the D2W total 
dose2_scale_factor = 1.0 #* 2e3 # Scale sim2 dose by this factor  (21993097--PBT_BRAIN - 3208725 primaries (2e3 less than required) )
slice_position = 0.5  # Fractional position (0-1) through volume for each slice
# =======================================

# Load images
print("Loading Sim1...")
print(sim1_ct)
ct1 = ft.readMHD(sim1_ct)
ct1_array = sitk.GetArrayFromImage(ct1)

print(ct1_array.max(), ct1_array.min())
# Handle single or multiple dose files

doses1 = [ft.readMHD(i) * dose1_scale_factor for i in sim1_doses]  # Read single file directly

print("\nLoading Sim2...")
ct2 = ft.readMHD(sim2_ct)
doses2 = [ft.readMHD(i) * dose2_scale_factor for i in sim2_doses]


"""
from here on in we have assumed weve sum all doses in the lists to create one dose, but i want to refactor to treat each dose separately, please save each with its field name too
"""




def resample_to_reference(image, reference, interpolator=sitk.sitkLinear, default_value=0.0):
    """Resample image to match reference image geometry."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(image)

# Visualization function
def plot_comparison(ct1, dose1, ct2, dose2, plane, slice_pos):
    """Plot side-by-side comparison for given slice orientation
    
    plane: 'XY' (axial), 'XZ' (coronal), or 'YZ' (sagittal)
    slice_pos: fractional position (0-1) through volume
    """
    
    # Resample sim2 to match sim1 spacing
    print(f"  Resampling sim2 to sim1 spacing for {plane} comparison...")
    ct2_resampled = resample_to_reference(ct2, ct1)
    dose2_resampled = resample_to_reference(dose2, dose1)
    dose1_resampled = dose1

    # Calculate point at slice position for CT1
    size1 = np.array(ct1.GetSize())
    spacing1 = np.array(ct1.GetSpacing())
    origin1 = np.array(ct1.GetOrigin())
    
    # Determine which axis to slice through based on plane
    if plane == 'XY':
        point1 = origin1 + [0, 0, size1[2] * spacing1[2] * slice_pos]
    elif plane == 'XZ':
        point1 = origin1 + [0, size1[1] * spacing1[1] * slice_pos, 0]
    elif plane == 'YZ':
        point1 = origin1 + [size1[0] * spacing1[0] * slice_pos, 0, 0]
    
    # Get slices manually for difference calculations
    ct1_slice = ft.getSlice(ct1, point=point1, plane=plane)
    dose1_slice = ft.getSlice(dose1, point=point1, plane=plane)
    ct2_slice = ft.getSlice(ct2_resampled, point=point1, plane=plane)
    dose2_slice = ft.getSlice(dose2_resampled, point=point1, plane=plane)
    
    # Calculate differences
    ct_diff = np.squeeze(sitk.GetArrayFromImage(ct1_slice) - sitk.GetArrayFromImage(ct2_slice))
    dose_diff = np.squeeze(sitk.GetArrayFromImage(dose1_slice) - sitk.GetArrayFromImage(dose2_slice))

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sim1 overlay - CORRECTED: ax first, then 3D images with point/plane
    ax = axes[0, 0]
    ft.showSlice(ax, imgBack=ct1, imgFront=dose1, point=point1, plane=plane, 
                 alphaFront=0.5)
    ax.set_title(f'Sim1: CT + Dose ({plane})')
    
    # Sim2 overlay (resampled) - CORRECTED
    ax = axes[0, 1]
    ft.showSlice(ax, imgBack=ct2_resampled, imgFront=dose2_resampled, 
                 point=point1, plane=plane, alphaFront=0.5)
    ax.set_title(f'Sim2: CT + Dose (resampled) ({plane})')
    
    # CT difference
    ax = axes[0, 2]
    im = ax.imshow(ct_diff, cmap='RdBu_r', vmin=-100, vmax=100)
    plt.colorbar(im, ax=ax, label='HU')
    ax.set_title(f'CT Difference (Sim1-Sim2)')
    ax.set_aspect('equal')
    
    # Dose1 - CORRECTED
    ax = axes[1, 0]
    ft.showSlice(ax, imgFront=dose1, point=point1, plane=plane)
    ax.set_title('Sim1 Dose')
    
    # Dose2 (resampled) - CORRECTED
    ax = axes[1, 1]
    ft.showSlice(ax, imgFront=dose2_resampled, point=point1, plane=plane)
    ax.set_title('Sim2 Dose (resampled)')
    
    # Dose difference
    ax = axes[1, 2]
    dose_max = max(np.abs(dose_diff).max(), 0.1)
    im = ax.imshow(dose_diff, cmap='RdBu_r', vmin=-dose_max, vmax=dose_max)
    plt.colorbar(im, ax=ax, label='Dose diff')
    ax.set_title(f'Dose Difference (Sim1-Sim2)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig, dose1_resampled, dose2_resampled


for field_dose in range(len(fields)):
    field_name  = fields[field_dose]

    dose1 = doses1[field_dose]
    dose2 = doses2[field_dose]

    # Print image information
    print("\n" + "="*60)
    print("SIM1 CT:")
    ft.displayImageInfo(ct1)
    print("\nSIM1 DOSE:")
    ft.displayImageInfo(dose1)

    print("\n" + "="*60)
    print("SIM2 CT:")
    ft.displayImageInfo(ct2)
    print("\nSIM2 DOSE:")
    ft.displayImageInfo(dose2)

    # Calculate differences
    print("\n" + "="*60)
    print("DIFFERENCES:")
    print(f"CT size match: {ct1.GetSize() == ct2.GetSize()}")
    print(f"CT spacing match: {np.allclose(ct1.GetSpacing(), ct2.GetSpacing(), rtol=0.01)}")
    print(f"Dose size match: {dose1.GetSize() == dose2.GetSize()}")
    print(f"Dose spacing match: {np.allclose(dose1.GetSpacing(), dose2.GetSpacing(), rtol=0.01)}")
    # Plot all three orientations
    print("\nGenerating visualizations...")
    fig_xy, _, _ = plot_comparison(ct1, dose1, ct2, dose2, plane='XY', slice_pos=slice_position)
    fig_xy.suptitle(f'Axial (XY) Slice Comparison - Dose 1 Scale Factor: {dose1_scale_factor}', fontsize=16, y=1.00)

    fig_xz, _, _ = plot_comparison(ct1, dose1, ct2, dose2, plane='XZ', slice_pos=slice_position)
    fig_xz.suptitle(f'Coronal (XZ) Slice Comparison - Dose 1 Scale Factor: {dose1_scale_factor}', fontsize=16, y=1.00)

    fig_yz, dose1_resampled, dose2_resampled = plot_comparison(ct1, dose1, ct2, dose2, plane='YZ', slice_pos=slice_position)
    fig_yz.suptitle(f'Sagittal (YZ) Slice Comparison - Dose 1 Scale Factor: {dose1_scale_factor}', fontsize=16, y=1.00)

    plt.show()
    
    ft.writeMHD(dose2_resampled, f"dose_to_water_total/gate_dose2W{field_name}.mhd")
    ft.writeMHD(dose1_resampled, f"dose_to_water_total/fred_dose2W{field_name}.mhd")
