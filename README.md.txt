how to run a simulation - YOU NEED A NVIDIA GPU ON WHATEVER MACHINE YOU'RE RUNNING ON:

0) ensure the venv is activated to access fred using correct python version
1) export dicom folder of the patient using the esapi script.
2) store the dicom folder in a directory inside /example_dcm. This path is called <path_to_dcm_path> henceforth.
3) if comparing to a known GATE dose, store the output dose .mhd and ct.mhd files in <path_to_dcm_path/output>.


4) to run a simulation, first we need to create the ct.mhd to be read by FRED using the dcm_to_mhd.py file
5) this python script takes the following args:
	parser.add_argument("--dcm_folder", "-d", default="dcm", help="Path to folder containing DICOM files")
    	parser.add_argument("--structure", "-s", help="Structure name to crop to")
    	parser.add_argument("--spacing", "-r", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Resample to spacing in mm (e.g., 2 2 2)")
    	parser.add_argument("--output", "-o", default="ct.mhd", help="Output CT filename (default: ct.mhd)")

We need to crop to the "Dose 0.1[%]" structure, as that is what GATE ct_cropped.mhd has done.

A typical command is <python dcm_to_mhd.py -d <path_to_dcm_path> -s "Dose 0.1[%]" -o CT.mhd


6) FRED has some in-built python scripts. E.g. to view a CT use <map_viewer -CT CT.mhd>
7) The main file describing the plan (gantry angles, couch angles, spot energies, spot positions) is the rtplan.inp

8) create this using the following:
Usage:
    python get_plan.py <dicom_folder> <beam_model_file> [output_file]

Example:
    python get_plan.py dicom_folder CustomBeamModel.bm rtplan.inp

9) here, we point the script to our exported dicom folder, we also show it the beam model to use, and we call the output rtplan.inp
10) now we have our CT.mhd, rtplan.inp files. We have everything ready to run fred.

11) run: fred -f fred.inp

12) okay. The dose is in out/Dose.mhd. But hold your horses cowboy, we need to do one more thing first.
13) you need to run compare_dose_and_ct.py as this produces an appropriate .mhd dose from our pre-calculated dose from GATE.
14) to do this, open that python file and find the following lines:

sim1_ct = "CT.mhd"
sim1_dose = ["out/Dose.mhd"]  # list format for consistency

sim2_ct = "example_dcm/dicom_data_1/output/ct_cropped.mhd"
sim2_dose = ["example_dcm/dicom_data_1/output/G130_T0_RS2_merged-Dose.mhd", 
             "example_dcm/dicom_data_1/output/G240_T0_RS2_merged-Dose.mhd"]  # multiple beams

sim1 is our fred output, sim2 is our gate output we have pasted into <path_to_dcm_path/output>.
change the path in the sim2_ct and sim2_dose to the correct path to the merged-Dose.mhd

Note: This bit could have been automated but I figure having some hands on keeps you engaged.

This does 2 things: Sums the dose from all separate fields, scales up by 2000x as we simulate 2000x fewer protons in GATE wrt required.

15) now run this with the following: python compare_dose_and_ct.py
This outputs a new .mhd of this sum and scaled GATE dose: "gate_d2w_sum.mhd". It also plots the two plan doses and CT side by side. Check these CT are the same.

16) Finally, run the next FRED ready script: <map_viewer -CT CT.mhd out/Dose.mhd gate_d2w_sum.mhd>
This runs the same in-built map_viewer script, passing in the CT FRED sees (should be nearly identical to ct_cropped.mhd from GATE, which you check 15) AND the two dose maps, FREDS dose, and the GATE dose we just summed and scaled.
17) Hopefully, you see nice agreement.









FULL EXAMPLE:


DICOM_DATA_PATH = <example_dcm/dicom_data_3>
GATE_OUTPUT_PATH = <example_dcm/dicom_data_3/output>

.\venv_fred\Scripts\activate

<navigate to working directory>

python dcm_to_mhd.py -d example_dcm/dicom_data_3 -s "Dose 0.1[%]" -o CT.mhd

### sortDicoms ###
# Found dicoms: 135 x CT, 1 x RS, 1 x RN, 2 x RD, 0 x PET, 0 x unknown
##################
Saved CT: CT.mhd

python .\get_plan.py example_dcm/dicom_data_3 CustomBeamModel.bm rtplan.inp

### sortDicoms ###
# Found dicoms: 135 x CT, 1 x RS, 1 x RN, 2 x RD, 0 x PET, 0 x unknown
##################
Reading beam model: CustomBeamModel.bm
Reading RN DICOM: example_dcm/dicom_data_3\2170542.dcm
  Table top positions (mm) - for reference only:
    Vertical: -155.639
    Longitudinal: 414.773999997734
    Lateral: 0.094999972888
  Isocenter (mm): [1.15318110405854, 21.8373976110776, 0.0]
  Plan: PBT Brain
  Patient:   Fields: 2
  Total spots: 1184
  Non-zero spots: 592
Generating .inp file sections...
Writing to: rtplan.inp
Done!

Successfully generated: rtplan.inp

fred -f fred.inp

Num of primary particles in primaryList: 5920000
Num primary   tot:      5920000
tracking  16.7 %
tracking 100.0 %

...

Number of primary particles:   5.92e+06
Tracking rate:   1.19e+06 primary/s
Track time per primary:    840.9 ns
################################################################################
Run with fred Version 3.70.0  (464eb1f) - 2024/06/02
################################################################################
Run wallclock time: 17 s

<< --------------------------------------------REMEMBER TO ALTER THE SIM_2 CT AND DOSE PATHS HERE IN THE compare_dose_and_ct.py SCRIPT ----------------------------------->>

python .\compare_dose_and_ct.py

Loading Sim1...
CT.mhd

Loading Sim2...

============================================================
DIFFERENCES:
CT size match: True
CT spacing match: True
Dose size match: False
Dose spacing match: False

Generating visualizations...
  Resampling sim2 to sim1 spacing for XY comparison...
  Resampling sim2 to sim1 spacing for XZ comparison...
  Resampling sim2 to sim1 spacing for YZ comparison...


map_viewer -CT CT.mhd out/Dose.mhd gate_d2w_sum.mhd
CT map loaded with shape (388, 199, 74)
['xy yz zx', 'xy', 'yz', 'zx']
[-9.057617 -3.289336 -6.5     ] [9.88769531 6.42746147 8.30000022]
Current position= [0.41503711 1.5690637  0.90000011]

Press return to redraw;  input "h" for help or "q" to exit: q







