import pandas as pd
import numpy as np
from pathlib import Path

folder = Path('Data/Input Data/diagnostic_tests')

df = pd.DataFrame(columns=['Battery Name', 'Cycle Number', 'Capacity', 'Discharge Capacity'])

for subfolder in folder.iterdir():
    x = subfolder.name.split("_")
    num = int(x[1])
    for file in subfolder.rglob('*.xlsx'):
        if file.is_file() and 'RPT' in file.name:     
            print(file)
            print(num)       
            bat = pd.read_excel(file, sheet_name = -1, skipfooter=1)
            cap = bat["Charge_Capacity(Ah)"].max()
            discap = bat["Discharge_Capacity(Ah)"].max()
            new_row = pd.DataFrame([{'Battery Name': file.name[9:11], 'Cycle Number': num
                                     , 'Capacity': cap, 'Discharge Capacity': discap
                                    }]
                                     )
            df = pd.concat([df, new_row], ignore_index=True)

print(df)

df.to_csv('Data/Intermediate Data/rawInfo2.csv', index=False)

# Note: For RPT 4 Data, the files were named slightly different from the other files, 
# so we manually changed Battery Name after running the extraction code. 