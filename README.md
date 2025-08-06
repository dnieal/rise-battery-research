# rise-battery-research

The global shift away from fossil fuels to reduce carbon emissions has intensified demand for lithium-ion batteries. Batteries considered to be “end-of-life” often retain sufficient residual capacity to be repurposed in “second-life” applications such as grid energy storage. In this study, we analyzed experimental data from Stanford’s Energy Control Lab, which simulated second-life grid storage conditions across nineteen discharge cycles for six electric vehicle (EV) batteries. We created a comprehensive analysis dataset by extracting important variables such as maximum discharge capacity, first-life operational characteristics, ambient temperature, and duty type from the raw data. We labeled each battery cycle to indicate whether their percentage capacity degradation exceeded the median value across all cycles. We developed and evaluated several machine learning models (logistic regression, random forest models, and KNN models) and utilized the Shapley method to identify key determinants of degradation. The logistic regression model had the highest predictive performance with 78% accuracy. Across all models, the ambient temperature and current number of cycles consistently were the primary predictors of battery health decline. The specific battery and past discharge capacity played a smaller but still significant role. Other factors such as past usage during the first life were found to have minimal impact on battery state of health. Our results offer important insights into the mechanisms governing second life lithium-ion battery degradation. We believe this research can be used to support more informed battery design and deployment strategies, contributing to longer battery life spans, cost effectiveness, and the advancement of a circular battery economy.


Keywords: Lithium-Ion Batteries, Second-Life Batteries, Circular Battery Economy, Maximum
Discharge Capacity, Renewable Energy, Sustainability

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

'''
Project/

├── README.md

├── Data/

│   ├── Analysis Data/

│      └── info2.csv

│      └── infoBatName.csv

│   ├── Input Data/

│      └── diagnostic tests/

│         ├── RPT_1/

│            └── Capacity_test_with_pulses

            ├──(other RPT folders…)

│         └── RPT_19/

│            └── Capacity_test_with_pulses

│   ├── Intermediate Data/

│      ├── info.csv

│      ├── rawInfo.csv

│      └── rawInfo2.csv

├── Output/

│   ├── Results/

│      └── Batteries Full Data Analysis.pdf

│      └──(individual pngs of all figures generated)

├── Scripts/

│   ├── Analysis Scripts/

│      └── knn.py

│      └── logreg.py

│      └── manualRemove.py

│      └── randfor.py

│      └── shapley.py


│   └── Processing Scripts/

│      ├── csvediting.py

│      ├── csvediting2.py

│      ├── extractRaw.py

│      └── finalcsv.py

'''



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
    - kaleido 0.4.0


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
