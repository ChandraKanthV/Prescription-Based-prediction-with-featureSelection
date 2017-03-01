import warnings
warnings.simplefilter('ignore')
import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
from IPython.display import display, HTML
import sklearn
from sklearn import tree
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfTransformer
import timeit


dataset_filename = '/home/chandra/Prescription-Based-prediction-with-featureSelection/roam_predcription_based_predicition_c.jsonl'
#dataset_filename ='/home/chandra/Prescription-Based-prediction-with-featureSelection/predBasedPres_corrected_50.jsonl'
dataframe_main = pd.DataFrame()
data = ''
with open(dataset_filename) as json_data:
        data = json.load(json_data)
        dataframe_main= json_normalize(data)
json_data.close()
#print "Finished reding the input file"
print dataframe_main.shape


column_names_list = dataframe_main.columns.values.tolist()
index_npi = column_names_list.index('npi')
column_names_list.remove('npi')

parts = [lable.strip(' \t\n\r').split('.',1) for lable in column_names_list] 
parts1 =  [parts_iter[0] for parts_iter in parts]
#len(parts)
column_name_new =  [parts_iter[1] for parts_iter in parts]
column_name_new.insert(index_npi,'npi')
#print len(column_name_new)
#print column_name_new


df_check = pd.DataFrame(parts)
print df_check.shape


df_names_check = pd.DataFrame(parts1, columns=['titles'])
counts = df_names_check.groupby(["titles"]).size()
#display (counts)

print "total number of rows is:", len(df_names_check['titles'])
print "cms_prescription_counts + provider_variables", counts['cms_prescription_counts']+counts['provider_variables']
print 'length of column_names_list', len(column_names_list)


column_name_new_striped = [x.strip(' \t\n\r') for x in column_name_new]
print len(column_name_new_striped)
#print column_name_new_striped

df_new_names_check = pd.DataFrame(column_name_new_striped, columns=['label']).groupby(['label']).size().reset_index()
df_new_names_check.columns = ['label', 'counts']

print df_new_names_check.mean()

column_names_list.insert(index_npi,'npi')

#dataframe_main.columns = column_name_new_striped
new_names_dict = {}
for i in range(0,len(column_names_list)):
    new_names_dict[column_names_list[i]] =  column_name_new_striped[i]
print len(new_names_dict)

#new_names_dict
dataframe_main.rename(columns=new_names_dict, inplace=True)

print "dataframe_main shape is:", dataframe_main.shape
#print "\n\ndataframe_main columns names after renameing:\n ", dataframe_main.columns

df_specialty_count = pd.DataFrame({'counts' : dataframe_main.groupby( [ "specialty"] , sort=False).size()}).reset_index()
df_specialty_count.sort_values(['counts'], ascending=False, inplace=True)
df_specialty_count.head()

print "\n\ndataframe_main columns names after renameing:\n ", dataframe_main.columns



dataframe_main.drop(['npi', 'brand_name_rx_count', 'gender', 'generic_rx_count', 'region', 'settlement_type', 'years_practicing'], axis=1,inplace=True)
#dataframe_main.columns.values[-10:]



#FOR THIS CAPSTONR PROJECT PURPOSE WE WILL CONSIDER ONLY TOP FIVE SPECIALITIES(WHICH CORESPONDS TO 2/3RD'S) FOR THE LACK OF COMPUTATIONAL INFRASTRUCTURE

tempList = []
df_df_specialty_countlessthan400 = df_specialty_count.query('counts < 5000')
for i in df_df_specialty_countlessthan400['specialty']:
    indices = dataframe_main.loc[dataframe_main['specialty'] == i].index.tolist()
    tempList.append(indices)  

#collect all the indices for these specialties to remove from the df_dataframe_main

indicesToDrop = [item for sublist in tempList for item in sublist]
dataframe_main.drop(dataframe_main.index[indicesToDrop], inplace=True)
dataframe_main.dropna(axis=1,how='all', inplace=True)
print "Shape of the data is:", dataframe_main.shape





label = dataframe_main.pop('specialty')
#dataframe_main.columns.values[-3:]

dataframe_main.fillna(value=0, inplace=True)

tfidf =TfidfTransformer(norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
data =tfidf.fit_transform(dataframe_main.values)

#dataframe_main.head(2)

#type (dataframe_main.values)



le = LabelEncoder()
y = le.fit_transform(label)

def featureSelectAndParamTuning(nmf_components):
	print "trying with NMF n_components=", 50*nmf_components
	model = NMF(n_components=50*nmf_components, init='random', random_state=0, verbose=0)
	boost_input = model.fit_transform(data, y=y)
	#print "Finished NMF"
	print "Shape of NMF outPut is:", boost_input.shape

	X_train,X_test,y_train,y_test = train_test_split(boost_input,y,test_size=0.2, random_state=42)

	model = XGBClassifier(silent=True,seed=42)
	param_grid = {
		#'subsample':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                #'scale_pos_weight':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                #'colsample_bytree':[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                'colsample_bylevel':[0.3,0.4,0.5,0.6,0.7,0.8,0.9]

              }


	start_time = timeit.default_timer()
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

#	for j in range(3,11):
#                print "started with colsample_bylevel:", j/10.0
#                start_time1 = timeit.default_timer()
#                param_grid = {
#                        'colsample_bylevel':[j/10.0]
#                }
#                grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold)
#                grid_result = grid_search.fit(X_train, y_train)
#                elapsed1 = timeit.default_timer() - start_time1
#                print " ",j/10.0, "took time :", elapsed1


	grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold)
	grid_result = grid_search.fit(X_train, y_train)

	# Get the estimator
	print grid_result.best_estimator_
	print 'CV Accuracy of best parameters: %3f' %grid_result.best_score_

	model = grid_result.best_estimator_
	y_pred = model.predict(X_test)

	print "Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(y_test, y_pred)
	#accuracy_score(y_test, y_pred)
	elapsed = timeit.default_timer() - start_time
	print elapsed

for i in range(1,11):
	featureSelectAndParamTuning(i)

