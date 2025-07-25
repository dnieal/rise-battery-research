# rise-battery-research

Welcome to our BU RISE Final Project, which focuses on the capacity of second-life lithium-ion batteries. 

## Section 1: Software and Platform

This project was developed and executed using:
- Python 3.10.14
- Jupyter Notebook 6.5.4

Required Python packages:
- pandas 2.2.0
- numpy 1.26.0
- matplotlib 3.8.0
- scikit-learn 1.4.1
- shap 0.45.1

No specialized hardware is required beyond a standard personal computer with at least 8 GB RAM and sufficient disk space (~500 MB). To run the scripts, ensure Python 3.10 and the above packages are installed. 

## Documentation Map

Project/
├── README.md
├── Data/
│   ├── Analysis Data/
│   │   └── info2.csv
│   ├── Input Data/
│   │   └── diagnostic tests/
│   │       ├── RPT_1/
│   │       │   └── Capacity_test_with_pulses
            ├──(other RPT folders…)
│   │       └── RPT_19/
│   │           └── Capacity_test_with_pulses
│   ├── Intermediate Data/
│   │   ├── info.csv
│   │   ├── rawInfo.csv
│   │   └── rawInfo2.csv
├── Scripts/
│   ├── Analysis Scripts/
│   │   └── logreg.py
│   └── Processing Scripts/
│       ├── csvediting.py
│       ├── csvediting2.py
│       ├── extractRaw.py
│       └── finalcsv.py



## Instructions for Reproducing Results

Follow the steps below to reproduce the results of this study.  
These steps assume that you have access to the Project/ folder and a computer with Python installed.

1. Download or clone the entire Project/ folder from the repository to your computer.

2. Ensure that Python 3.10 (or later) is installed on your system.
   - If Python is not installed, download it from https://www.python.org/downloads/.

3. Install the required Python packages listed in Section 1 of the README:
    - pandas 2.2.0
    - numpy 1.26.0
    - matplotlib 3.8.0
    - scikit-learn 1.4.1
    - shap 0.45.1


4. Prepare the data:
   - The original raw datasets are stored in the Input Data/diagnostic tests folder.
   -    The processing scripts will process these raw files and construct a new dataset.
   -        Note: csvediting.py was written to format the rawData from cycles 1-16
   -        Note: csvediting2.py was written to format the rawData from cycles 17-19 because the formats for cycles 1-16 and cycles 17-19 are slightly different. 
   -        Note: for RPT 4 Data, the files were named slightly different from the other files, so we manually changed Battery Name after running the extraction code. 

   - Alternatively, navigate to info2.csv under Analysis Data to directly access our constructed dataset. 
       

5. Run the analysis:
   - The main analysis script is logreg.py underneath the Analysis Scripts Folder. 
   - If executed, this script will load the constructed dataset and perform the logistic regression as well as the Shapley value method.
   - *ADD ONCE VISUALIZATION CODE CREATED*

6. Verify outputs:
   - After running the scripts, check the Output/ folder:
       - Output/Tables/ will contain CSV files summarizing results.
       - Output/Figures/ will contain image files (e.g., .png) of plots and visualizations.


Following these steps from start to finish will reproduce the results exactly as they are presented in the Output/ folder of this project.
