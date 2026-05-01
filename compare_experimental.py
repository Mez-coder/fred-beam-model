import pandas as pd
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import fredtools as ft

def get_experimental_sigmas_air(energy):
    df = pd.read_excel("UCLH_Spot_Profiles_All_Sigmas.xlsx", sheet_name="Sheet3")
    distances = np.array([-40, -20, -10, 0, 10, 20, 40])
    row = df[df["Energy"] == energy]
    row = row.iloc[0]
    sigma_x = [row[f"sigmaX_{d}"] for d in distances]
    sigma_y = [row[f"sigmaY_{d}"] for d in distances]
    return sigma_x, sigma_y

def get_experimental_IDD_water(energy):
    df = pd.read_excel("UCLH_reference_IDD.xlsx", sheet_name=str(energy))
    x = df['Depth (mm)']
    y = df['Dose (normalised to 1.0 at 20mm deep)']
    return x, y

def get_fred_IDD(energy, norm_depth=20):
    dose_img = ft.readMHD(f"out_{energy:.1f}MeV/Dose.mhd")
    dose_array = sitk.GetArrayFromImage(dose_img)
    dx, dy, dz = dose_img.GetSpacing()
    nz, ny, nx = dose_array.shape
    depths = np.arange(ny) * dy
    idd_raw = np.sum(dose_array, axis=(0, 2))
    dose_at_norm = float(np.interp(norm_depth, depths, idd_raw))
    idd_norm = idd_raw / dose_at_norm
    return depths, idd_norm

ENERGIES = [80, 100, 120, 140, 160, 180, 200]
for energy in ENERGIES:
    x_exp, y_exp = get_experimental_IDD_water(energy)
    x_fred, y_fred = get_fred_IDD(energy)
    plt.figure()
    plt.plot(x_exp, y_exp, label='Experimental')
    plt.plot(x_fred, y_fred, label='FRED')
    plt.legend()
    # plt.xlim(0, 100)
    plt.xlabel('Depth (mm)')
    plt.ylabel('IDD (normalised to 1.0 at 20mm deep)')
    plt.title(f'{energy} MeV beam', fontweight='bold')
    plt.show()
