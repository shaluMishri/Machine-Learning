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
df.rename(columns={'Segment':'Tag_Name'}, inplace=True)
header = ['Tag_Name','Biz Term Group','Data Domain Group','Parent Data Domain','Data Domain','Business Term']
df.to_csv(pathToSave, columns = header,sep=',',encoding='utf-8',index=False)