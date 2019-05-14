# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:58:04 2019

@author: Admin
"""

import pandas as pd
import numpy as np
# reading the csv file 
housing = pd.read_csv('housing.csv') 
#to see the first 5 elements of the table 
housing.head()
# gives the detailed info of the table 
housing.info()
# gives the mean median mode of the quantitative attributes 
housing_describe = housing.describe() 
#In a given column It gives the number of unique elements along with its count 
housing['ocean_proximity'].value_counts()

# plotting histogram to get the overall picture of the data 
import matplotlib.pyplot as plt 
housing.hist(bins=50,figsize=(20,20)) # number of bins is 50 in this case 

# splitting the train test data 
from sklearn.model_selection import train_test_split 
train_set,test_set = train_test_split(housing,test_size=0.2,random_state= 42)

housing["income_cat"]=np.ceil(housing["median_income"]/1.5)
housing.hist(column='income_cat')

#Doubt?
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)

housing["income_cat"].value_counts()/len(housing)

from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
    

for set_ in(strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)
    
housing = strat_train_set.copy()

housing.plot(kind="scatter",x="longitude",y="latitude")
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)

#Why X axis label is not printing?
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
             s=housing["population"]/100,label="population",figsize=(10,7),
             c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,)
#s = size
#c = color
#jet = jet returns the jet colormap as a three-column array

#why legend?
plt.legend()


corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.tools.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms",
            "housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))

housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)

housing["rooms_per_hopusehold"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing=strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set["median_house_value"].copy()

housing.dropna(subset=["total_bedrooms"])
housing.drop("total_bedrooms",axis=1)
median=housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True)

from sklearn.preprocessing import Imputer
imputer=Imputer(strategy="median")
housing_num=housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)

imputer.statistics_                                                                                                                      
housing_num.median().values

X=imputer.transform(housing_num)
housing_tr=pd.DataFrame(X,columns=housing_num.columns)

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
housing_cat=housing["ocean_proximity"]
housing_cat_encoded=encoder.fit_transform(housing_cat)
housing_cat_encoded

print(encoder.classes_)

from sklearn.base import BaseEstimator,TransformerMixin
rooms_ix,bedrooms_ix,population_ix,household_ix=3,4,5,6
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household=X[:,rooms_ix]/X[:,household_ix]
        population_per_household=X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,household_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]
        
attr_adder=CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs=attr_adder.transform(housing.values)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline=Pipeline([('imputer',Imputer(strategy="median")),('attribs_adder',
             CombinedAttributesAdder()),('std_scaler',
             StandardScaler()),])
housing_num_tr=num_pipeline.fit_transform(housing_num)

from sklearn.base import BaseEstimator , TransformerMixin 
class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values
    
from sklearn.preprocessing import LabelBinarizer    
num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]
num_pipeline=Pipeline([('selector',DataFrameSelector(num_attribs)),
                       ('imputer',Imputer(strategy="median")),
                       ('attribs_adder',CombinedAttributesAdder()),
                       ('std_scaler',StandardScaler()),])
cat_pipeline=Pipeline([('selector',DataFrameSelector(cat_attribs)),
                       ('label_binarizer',LabelBinarizer()),])
    
from sklearn.pipeline import FeatureUnion
full_pipeline=FeatureUnion(transformer_list=[("num_pipeline",num_pipeline),
                                             ("cat_pipeline",cat_pipeline),])  
    
housing_prepared=full_pipeline.fit_transform(housing)
housing_prepared

from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared , housing_labels)
some_data = housing.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))

from sklearn.metrics import mean_squared_error
housing_predictions= lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_rmse
 
#Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)

housing_predictions=tree_reg.predict(housing_prepared)
tree_mse=mean_squared_error(housing_labels,housing_predictions)
tree_rmse=np.sqrt(tree_mse)
tree_rmse

#cross validation
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores) 

from sklearn.model_selection import cross_val_score
scores=cross_val_score(tree_reg,housing_prepared,housing_labels,
                       scoring="neg_mean_squared_error",cv=10) #Doubt?
tree_rmse_scores=np.sqrt(-scores)

def display_scores(scores):
    print("scores:",scores)
    print("Mean",scores.mean())
    print("Standard deviation:",scores.std())
display_scores(tree_rmse_scores) 

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

display_scores(forest_rmse_scores)


from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl")
my_model_loaded = joblib.load("my_model.pkl")



  















    
    
















