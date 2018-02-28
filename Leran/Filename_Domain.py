import sys
import pandas as pd
"""
Python Script to Save Tags in csv format, Script take 2 args 
   Args1 pathToRead = path to read the file 
   Args2 pathToSave = path to save in csv file

"""


pathToRead = sys.argv[-2]
pathToSave = sys.argv[-1]

xl = pd.ExcelFile(pathToRead)
df = xl.parse(xl.sheet_names[0])
arr = df['Segment'].unique()
df_new = pd.DataFrame(arr,columns=['Segment'])
df_new['Domain Name'] = df_new.Segment.apply(lambda row: (row + ' Domain'))
df_new.to_csv(pathToSave, sep=',',index=False)