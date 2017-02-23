#!/usr/bin/python
import time
import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import os
from copy import copy
#from update import update

dataset_filename = '/home/chandrakanth/WORK/Kaggle/roam_predcription_based_predicition.jsonl'
dataset_filename_1 = '/home/chandrakanth/WORK/Kaggle/roam_prescription_based_predicition_small_1.jsonl'
dataset_filename_2 = '/home/chandrakanth/WORK/Kaggle/roam_prescription_based_predicition_small_2.jsonl'
dataset_filename_10 = '/home/chandrakanth/WORK/Kaggle/roam_prescription_based_predicition_small_10.jsonl'
dataset_filename_c = '/home/chandrakanth/WORK/Kaggle/predBasedPres_corrected.jsonl'


columns_b = ['npi','gender', 'specialty', 'region', 'settlement_type' , 'years_practicing' , 'brand_name_rx_count', 'generic_rx_count']
dataframe_collection = {}

for i in range(0,24):
	dataset_filename = '/home/chandrakanth/WORK/Kaggle/Prescription_Based_Prediction/inputFiles1/x'+"%02d"%i+'.jsonl'
	#print dataset_filename
	dataframe_collection[i] = pd.DataFrame(columns=columns_b)
	localtime = time.asctime( time.localtime(time.time()) )
#	print "Local current time :", localtime
#	with open(dataset_filename, 'rt') as f:
#		j=0
#		for line in f:
#			ex = json.loads(line)
#			npi = {'npi':ex['npi']}
#			pv = ex['provider_variables']
#			cpc = ex['cms_prescription_counts']
#			con = {}
#			con.update(npi)
#			con.update(pv)
#			con.update(cpc)
#			column_list = columns_b + list(cpc)
#			df_iter = pd.DataFrame(con, index=[1], columns=column_list)    
#			dataframe_collection[i] = dataframe_collection[i].append(df_iter, ignore_index = True)
#			print j
#			j=j+1
#		f.close()
#		print i,dataframe_collection[i].shape

	data = ''
	print i, 'opening file dataset_filename'
	with open(dataset_filename) as json_data:
		data = json.load(json_data)
		dataframe_collection[i] = json_normalize(data)
	json_data.close()
	print i, 'closeing file dataset_filename'


	localtime = time.asctime( time.localtime(time.time()) )
	print "Local current time :", localtime

dataframe_main = pd.DataFrame(columns=columns_b)
#for i in range(0,24):
#	print i
#	dataframe_main = dataframe_main.append(dataframe_collection[i], ignore_index = True)


print 1
dataframe_temp0001 = dataframe_collection[0].append(dataframe_collection[1], ignore_index = True)
print 2
dataframe_temp0203 = dataframe_collection[2].append(dataframe_collection[3], ignore_index = True)
print 3
dataframe_temp0405 = dataframe_collection[4].append(dataframe_collection[5], ignore_index = True)
print 4
dataframe_temp0607 = dataframe_collection[6].append(dataframe_collection[7], ignore_index = True)
print 5
dataframe_temp0809 = dataframe_collection[8].append(dataframe_collection[9], ignore_index = True)
print 6
dataframe_temp1011 = dataframe_collection[10].append(dataframe_collection[11], ignore_index = True)
print 7
dataframe_temp1213 = dataframe_collection[12].append(dataframe_collection[13], ignore_index = True)
print 8
dataframe_temp1415 = dataframe_collection[14].append(dataframe_collection[15], ignore_index = True)
print 9
dataframe_temp1617 = dataframe_collection[16].append(dataframe_collection[17], ignore_index = True)
print 10
dataframe_temp1819 = dataframe_collection[18].append(dataframe_collection[19], ignore_index = True)
print 11
dataframe_temp2021 = dataframe_collection[20].append(dataframe_collection[21], ignore_index = True)
print 12
dataframe_temp2223 = dataframe_collection[22].append(dataframe_collection[23], ignore_index = True)
print 13

for ii in range(0,24):
	del dataframe_collection[ii]

dataframe_temp00010203 = dataframe_temp0001.append(dataframe_temp0203, ignore_index = True)
print 14
dataframe_temp04050607 = dataframe_temp0405.append(dataframe_temp0607, ignore_index = True)
print 15
dataframe_temp08091011 = dataframe_temp0809.append(dataframe_temp1011, ignore_index = True)
print 16
dataframe_temp12131415 = dataframe_temp1213.append(dataframe_temp1415, ignore_index = True)
print 17
dataframe_temp16171819 = dataframe_temp1617.append(dataframe_temp1819, ignore_index = True)
print 18
dataframe_temp20212223 = dataframe_temp2021.append(dataframe_temp2223, ignore_index = True)
print 19

del dataframe_temp0001
del dataframe_temp0203
del dataframe_temp0405
del dataframe_temp0607
del dataframe_temp0809
del dataframe_temp1011
del dataframe_temp1213
del dataframe_temp1415
del dataframe_temp1617
del dataframe_temp1819
del dataframe_temp2021
del dataframe_temp2223


dataframe_temp_3_1 = dataframe_temp00010203.append(dataframe_temp04050607, ignore_index = True)
print 20
dataframe_temp_3_2 = dataframe_temp08091011.append(dataframe_temp12131415, ignore_index = True)
print 21
dataframe_temp_3_3 = dataframe_temp16171819.append(dataframe_temp20212223, ignore_index = True)
print 22

del dataframe_temp00010203
del dataframe_temp04050607
del dataframe_temp08091011
del dataframe_temp12131415
del dataframe_temp16171819
del dataframe_temp20212223

print 23
dataframe_temp_4_1 = dataframe_temp_3_1.append(dataframe_temp_3_2, ignore_index = True)
print 24
dataframe_fininal = dataframe_temp_4_1.append(dataframe_temp_3_3, ignore_index = True)
print 25
del dataframe_temp_3_1
del dataframe_temp_3_2
del dataframe_temp_3_3

del dataframe_temp_4_1

print dataframe_fininal.shape

