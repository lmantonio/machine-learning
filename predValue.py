import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
import matplotlib.tri as tri
import matplotlib.style
import matplotlib as mpl
import os #used to save the plot in the designated directory
import xlsxwriter

workbook = xlsxwriter.Workbook('solar.xlsx')
worksheet = workbook.add_worksheet('model')

worksheet.write('A2', 'Hello world2222')

worksheet2 = workbook.get_worksheet_by_name('model')
worksheet2.write_row(0,0,'13')

workbook.close()

'''
f = open('modelcoef.txt')
line = f.readline()  
line_str = line.split()     
numCoef = len(line_str)
f.close()

modelCoef = np.zeros((numCoef,1))

print(line_str)

for i in range(0,numCoef):
    modelCoef[i][0] = float(line_str[i])

print('Insert', numCoef-1 ,'feature(s) value(s):')
user_data = input('\n')
print("\n")
user_list = user_data.split()

if (len(user_list)!=numCoef-1):
    print('Error: number of features does not match with the model')
else:
    x = np.zeros((len(user_list)+1,1))
    x[0] = 1.0

    for i in range(1,len(user_list)+1):
        x[i][0] = float(user_list[i-1])        
    
    y = np.matmul(x.transpose(),modelCoef)

    print('Model predicition: R$', y[0][0])
'''