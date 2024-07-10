#################       To get the encoding of a data frame using CHARDET      ##########################
import chardet
with open('filename.csv', 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))

print(result)


#########################       Steps in data analysis      ##########################################################
1- Combine the train and test data together
2- Check the data description for NA to know which value to be filled with None or mode(SimpleImputer(most_frequent))
3- Fill the numerical features with KNNimputer
4- Check the numerical variable to see if it is skewed and apply np.log1p transformation to it
5- Apply chi-squared to the categorical features and correlation to the numerical features and keep the features apart
6- Transform the target variable using np.log(y) for training but use np.exp(y_pred) to get the actual result
7- Perform Column Transformer(Standard scaler/OneHotEncoding) on the all_data with/without the correlated features and chi-squared features



for i in ['list of columns in the data description that are described as NA']:
    all_data[i].fillna('None', inplace=True)

import warnings      # to stop displaying warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500) #to show maximum number of columns in dataframe

#remove data that are highly correlated with themselves
plt.figure(figsize=(12, 12))
sns.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()


df.isnull().sum()  #to check for missing values

(df.isnull().sum()/df.shape[0]) * 100   #to get the percentage of missing values. if about 50% missing value is detected, delete the column

df.duplicated().sum()         #to check for duplicates | use df.drop_duplicates(inplace=True) to remove the duplicates



################    To split a dataframe into train/test set based on the most correlated feature(stratified sampling)  ############
df['cat_feature'] = pd.cut(df['continuous_feature'],bins=[0.4,2.2,4,5.8,np.inf], labels=[1,2,3,4]) #use the min,25%,50%,75% for the bins

df['cat_feature'].hist()  #to visulaize the ditribution contrast to the original distribution

from sklearn.model_selection import StratifiedShuffleSplit  #it is better than train_test_split(random sampling)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["cat_feature"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index] 
    
#After splitting, drop the cat_feature from the train and test dataset 


#################### To fill a numeric variable grouped by a categorical variable  ########################

df['numeric_feature'] = df.groupby('categorical_feature')['numeric_feature'].transform(lambda x: x.fillna(x.median()))

for col in df.columns:          #to fill missing values in the features with the mean
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

for i in df.select_dtypes('object').columns:   #to check for garbage values in the strings(object)
    print(df[i].value_counts())
    print('****'*10)
    

for i in df.select_dtypes(include='number').columns:   #to display histogram of all the numerical features
    sns.histplot(data=df,x=i)
    plt.show()
    
df.hist(bins=20, figsize=(10, 10))  #A simplified way of viewing the histogram
plt.show()

    
for i in df.select_dtypes(include='number').columns:  #to display boxplot to detect outliers
    sns.boxplot(data=df,x=i)
    plt.show()
    
for i in df.select_dtypes(include='number').columns:   #to draw the scatter plot for each feature with the target
    sns.scatterplot(data=df,x=i,y='target')
    plt.show()
    
from sklearn.impute import KNNImputer    #to fill-in missing values using knnimputer for only numerical features 
imputer = KNNImputer()
for i in df.select_dtypes(include='number'):
    df[i] = imputer.fit_transform(df[[i]])
    

    
from sklearn.impute import SimpleImputer    #Using SimpleImputer to fill missing values in categorical features
imputer = SimpleImputer(strategy='most_frequent')
for i in df.select_dtypes(include='object'):
    df[i] = imputer.fit_transform(df[[i]])
    
    

#DO NOT DO THE OUTLIER TREATMENT FOR DISCRETE VARIABLES,ONLY CONTINOUS FEATURES
#YOU DON'T NEED TO DO THE OUTLIER TREATMENT FOR EVERY VARIABLES


############################# To remove outliers using Z-score emperical rule 68-95-99.7% (best method) ###################################
def detect_outliers(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    outliers = []
    result = []
    
    for i in data:
        z_score = (i - mean)/std
        if np.abs(z_score) > threshold:
            outliers.append(i)
        else:
            result.append(i)
        
    return result
    
for i in df.select_dtypes('number').columns:
    df[i] = pd.Series(detect_outliers(df[i]))

#to confirm that the outliers have been removed
for i in df.select_dtypes('number').columns:  
    sns.boxplot(df[i])
    plt.show()
    
df = df.dropna()  #to remove the outliers that are now null values

#############################  steps to remove the outliers using IQR   ##########################################
def whisker(col):
    q1,q3 = np.percentile(col,[25,75])
    iqr = q3 - q1
    lw = q1 - 1.5*iqr
    uw = q3 + 1.5*iqr
    return lw,uw
    
for i in ['f1','f2','f3',...,'fn']:
    lw,uw = whisker(df[i])
    df[i] = np.where(df[i] < lw, lw, df[i])
    df[i] = np.where(df[i] > uw, uw, df[i])
 



 for i in ['f1','f2','f3',...,'fn']:
    sns.boxplot(df[i])
    plt.show()
    

#To convert categorical variables into numerical variables
df_new = pd.get_dummies(data=df,columns=df.select_dtypes(include='object').columns,drop_first=True)



df.dropna(subset='feature name',inplace=True)  # to drop all rows in dataframe with missing values in the feature name
    

######################################## groupby functions #################################################
df.groupby('key').aggregate(['min','median','max'])  #aggregate functions of groupby

df.groupby(['f1', 'f2']).aggregate({'f3':'mean','f4':'mean' })

df.groupby('f1').first() #it returns a dataframe with the first item in the group
df.groupby('f1').last() #it returns a dataframe with the last item in the group

df.groupby('key').filter(lambda x: x['data2'].std() > 4)  #that x is the data frame 1.e the filter function returns a df

df.apply(lambda x: x.mean())  #it applies the mean function to each columns by default, if axis=1, it applies to each rows
df.groupby('key')['speed_mph'].aggregate(['mean','count']).query('count > 10').sort_values('mean',ascending=False).plot(kind='barh') #trick

df['x_cat'] = pd.qcut(df['x'],4)  #to split a series into quantiles (the number can vary)

#sigma-clipping method of removing outliers
quartiles = np.percentile(df['feature'], [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])
df = df.query('(feature > @mu - 5 * @sig) & (feature < @mu + 5 * @sig)')



###################################      Aggregate(min,max,mean) Statistics on a data frame       ####################################

#id is a categorical feature used in groupby
stat_ = ['max', 'min', 'mean']
df_attr = ['f1','f2','f3','f4','f5','f6']
df_stat = df.groupby(['id']).agg({'f1': stat_, 'f2': stat_, 'f3': stat_, 'f4': stat_, 'f5': stat_, 'f6': stat_,})

df_stat.columns = ['_'.join(x) for x in zip(df_stat.columns.get_level_values(0), df_stat.columns.get_level_values(1))]
df_stat = df_stat.reset_index()

df_stat.head()  #to check our data

### to calculate difference between the df_stat maximum and minumum features
for attr in df_attr:
    df_stat[f'diff_max_min_{attr}'] = df_stat[f'{attr}_max'] - df_stat[f'{attr}_min']
    
df_stat.head()  #to check our data


### To subtract the first item in the id(groupby) from the mean features...the first could be january
for attr in df_attr:
    df_stat[f'diff_first_mean_{attr}'] = df[df['id'].isin(df_stat['id'])].groupby(['id'])[attr].nth(0).values - df_stat[f'{attr}_mean']
    
df_stat.head()

### To subtract the last item in the id(groupby) from the mean features... the last could be december
for attr in df_attr:
    df_stat[f'diff_last_mean_{attr}'] = df[df['id'].isin(df_stat['id'])].groupby(['id'])[attr].nth(-1).values - df_stat[f'{attr}_mean']
    
df_stat.head()


### To subtract the the january and december value from each other in the id(groupby) ordered by date
for attr in ['f1','f2','f3','f4','f5','f6']:
    diff_dec_jan_temp = df.sort_values(by=['date']).groupby(['id'])[attr].nth(-1) - df.sort_values(by=['date']).groupby(['id'])[attr].nth(0)
    diff_dec_jan_temp = diff_dec_jan_temp.reset_index(name=f'diff_dec_jan_{attr}')
    df = df.merge(diff_dec_jan_temp, on='id',how='left') #to append the feature to the dataframe


'''you can now merge it to another dataframe'''

###################  To perform chi-squared test on the categorical features against the target to check importance ###############
# the X variable comprises of categorical features only
# the y is the target variable
# 0.5 is the confidence interval

from sklearn.feature_selection import chi2
chi_scores = chi2(X,y) #the first index is the chi_scores, the second index is the p-values

#the lower the p_values < 0.5, the higher the importance
p_values = pd.Series(chi_scores[1],index=X.columns)
p_values.sort_values(ascending=False,inplace=True)
p_values.plot.bar()


####################    LABEL ENCODING and MUTUAL INFORMATION    ###########################################
#first split the data frame into X and y

#label encoding
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()
    

#mutual information
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
    
    
################################ PREPROCESSING OF TRAIN AND TEST DATA  ##################################################
#df is the train data
#test_df is the test data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

X = df.drop('target_feature', axis=1)
y = df.target_feature
all_data = pd.concat([X, test_df])

num_attribs = []
discrete = []
for i in all_data.select_dtypes(include='number').columns:
    if(all_data[i].nunique() <= 25):
        discrete.append(i)
    else:
        num_attribs.append(i)     
        
cat_attribs = discrete + list(all_data.select_dtypes(include='object').columns)

num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
    ])

full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
     ])

all_data_prepared = full_pipeline.fit_transform(all_data)

train_processed = full_pipeline.transform(X)
train_processed = pd.DataFrame(train_processed.toarray())

test_processed = full_pipeline.transform(test_df)
test_processed = pd.DataFrame(test_processed.toarray())



#############     PREPROCESSING OF TRAIN AND TEST DATA Using Combined Attribute Adder (Feature Engineering)   #########################
#After doing EDA to get the best features that can correlate with the target feature, then you tweak the feature engineering

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


feature1, feature2, feature3  = 3, 4, 5  # index of each features... index starts from 0

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_feature1_per_feature2 = True): 
        self.add_feature1_per_feature2 = add_feature1_per_feature2  #tweaking parameter
        
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        feature1_per_feature3 = X[:,  feature1] / X[:, feature3]
        feature3_per_feature2 = X[:,feature3] / X[:,feature2]
        
            if self.add_feature1_per_feature2:
            feature1_per_feature2 = X[:,  feature1] / X[:, feature2]
            return np.c_[X, feature1_per_feature3, feature3_per_feature2, feature1_per_feature2]
        else:
            return np.c_[X, feature1_per_feature3, feature3_per_feature2]


X = df.drop('target_feature', axis=1)
y = df.target_feature

all_data = pd.concat([X, test_df])        

num_attribs = []
discrete = []
for i in all_data.select_dtypes(include='number').columns:
    if(all_data[i].nunique() <= 25):
        discrete.append(i)
    else:
        num_attribs.append(i)     
        
cat_attribs = discrete + list(all_data.select_dtypes(include='object').columns)

num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder(add_feature1_per_feature2=False)),  #it can be False or True
            ('std_scaler', StandardScaler()),
    ])


full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
     ])


all_data_prepared = full_pipeline.fit_transform(all_data)

train_processed = full_pipeline.transform(X)
train_processed = pd.DataFrame(train_processed.toarray())

test_processed = full_pipeline.transform(test_df)
test_processed = pd.DataFrame(test_processed.toarray())



########## To check for feature importance using Random Forest  model
cat_one_hot_attribs = list(full_pipeline.named_transformers_['cat'].categories_[0])
extra_attribs = ['feature1_per_feature3', 'feature3_per_feature2', 'feature1_per_feature2']
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(model.feature_importances_, attributes), reverse=True)


#########################     To calculate the Confidence Interval on the final Predictions    ###########################################

from scipy import stats
confidence = 0.95
squared_errors = (y_test - y_pred) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))

######################## Special way of performing preprocessing when using LabelEncoder  ################################
from sklearn.preprocessing import StandardScaler
for i in num_attribs:
    scaler = StandardScaler()
    X[i] = scaler.fit_transform(X[[i]])
    
    
from sklearn.preprocessing import LabelEncoder
for i in cat_attribs:
    label_encoder = LabelEncoder()
    X[i] = label_encoder.fit_transform(X[i])
    print(f'mappings for {i}: ', {index:value for index,value in enumerate(label_encoder.classes_) })
    print()



######################### To check for SKEWED data in the numerical feature #############################################
#If the Absolute Skew is greater than 0.5, then it is skewed
import scipy
from scipy import stats
from scipy.stats import norm

skewed_df = pd.DataFrame(num_attribs, columns=['Feature'])
skews = []
for i in num_attribs:
    skews.append(stats.skew(all_data[i]))
skewed_df['Skew'] = skews
skewed_df['Absolute'] = np.abs(skews)

skewed_features = list(skewed_df.query('Absolute > 0.5')['Feature'])
skewed_features


###############  To normalize all the skewed_features using np.log1p() into a noraml distribution ##########################        
for col in skewed_features:
    all_data[col] = np.log1p(all_data[col])
        
# To visualize the distribution of the transformed data against a normal distribution 
for i in skewed_features:
    sns.distplot(all_data[all_data[i] > 0][i], fit=norm)
    plt.show()
    

#################################### Times series stock analysis  ####################################

df.reset_index(drop=True, inplace=True) #to reset index 

df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')  #to convert a feature to a real date

df['date feature'].dt.  #to get date,year,month of a datetime feature

import datetime      #basic way of creating a date timestamp
date = datetime.datetime(2024,5,25)

from dateutil import parser    #to load date from a string
date = parser.parse('27th march 1995')

date.strftime('%A') #to get the day of the week like monday,tuesday...

df['feature'].plot()
df['feature'].resample('BA').mean().plot()  #the resample('BA') get the mean of the feature for a complete year, 'd' is for daily
df['feature'].rolling(30).sum().plot()  #the rolling(30) refers to the sum of 30 days, if it is 365, then sum of 365 days

############################3 To add Days of Week as features to a time series dataframe ###################################
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    df[days[i]] = (df.index.dayofweek == i).astype(float)
    

#########################   to split time dataframe index based on weekday and weekend to determine the trends ##########################
new_df = df.groupby([np.where(df.index.weekday < 5 , 'weekday', 'weekend'),df.index.time]).mean()
new_df.loc['weekday':'weekday'].plot()
new_df.loc['weekend':'weekend'].plot()



##################################################    PYCARET        ################################################################
#export the dataframe to google colab
# !pip install pycaret
import pycaret
from pycaret.regression import setup, compare_models
_ = setup(data=df, target='the target name in the df')
compare_models()

from sklearn.linear_model import BayesianRidge,Ridge,OrthogonalMatchingPursuit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


#After training, select the top best models and train it on the dataframe and get the results
models = {
    "br": BayesianRidge(),
    "lightgbm": LGBMRegressor(),
    "ridge": Ridge(),
    "omp": OrthogonalMatchingPursuit()
}

for name, model in models.items():
    model.fit(X_train,y_transformed)
    

results = {}
for name, model in models.items():
    result = np.exp(np.sqrt(-cross_val_score(model, X_train, y_transformed, scoring='neg_mean_squared_error', cv=10)))
    results[name] = result
    
### check for the one that has lowest mean and variance for a regression task and use it to get the predictions
for name, result in results.items():
    print("----------\n" + name)
    print(np.mean(result))
    print(np.std(result))
    
    
y_pred = (
    0.35 * np.exp(models['br'].predict(X_test)) +
    0.2 * np.exp(models['lightgbm'].predict(X_test)) +
    0.3 * np.exp(models['ridge'].predict(X_test)) +
    0.15 * np.exp(models['omp'].predict(X_test))
)


#####################################           To Save A Model        #############################################################

from sklearn.externals import joblib

joblib.dump(my_model, "my_model.pkl")
my_model_loaded = joblib.load("my_model.pkl")


#################################      A Fake Classifier that does nothing     ###############################################

from sklearn.base import BaseEstimator

class FakeClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

   
 #####################################    Recall, Precision, F1-Score     #######################################################
#The lower the threshold, the lower the precison and the higher the recall
#The model uses decison_function/predict_proba instead of predict to get the threshold

y_score = model.descison_function(X)  #to get the threshold... model.prdict_proba(X)
y_scores = cross_val_predict(model,X,y,method='decision_function')  #to get thresholds
y_scores = cross_val_predict(model,X,y,method='predict_proba')  #to get thresholds using RandomForest

################################      Visualization of  Precision_Recall_Thresholds Curve    ###########################################
from sklearn.metrics import precision_recall_curve, classification_report

y_scores = cross_val_predict(model,X_train,y_train,method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

plt.figure(figsize=(15,8))
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.legend()

#let's assume i want a 90% precision, what recall will i get????
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
threshold_90_precision  #threshold that corresponds to the 90% precison

y_pred_90 = y_scores >= threshold_90_precision

print(classification_report(y_train,y_pred_90))


################################      Visualization of  ROC_AUC Curve( Binary Classifier)   ###########################################
#let's draw the ROC_AUC curve for four  models

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

y_scores_forest = cross_val_predict(RandomForestClassifier(), X_train,y_train, cv=5, method='predict_proba')
y_scores_forest = y_scores_forest[:,1] # it return negative and positive classes

y_scores_sgd = cross_val_predict(SGDClassifier(), X_train,y_train, cv=5, method='decision_function')

y_scores_logistic = cross_val_predict(LogisticRegression(), X_train,y_train, cv=5, method='decision_function')

y_scores_svc = cross_val_predict(SVC(), X_train,y_train, cv=5, method='decision_function')

from sklearn.metrics import roc_curve, roc_auc_score
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, y_scores_forest)
fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train, y_scores_sgd)
fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(y_train, y_scores_logistic)
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_train, y_scores_svc)

def plot_roc_curve(fpr, tpr, label):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    

plot_roc_curve(fpr_forest, tpr_forest, label='Random Forest')
plot_roc_curve(fpr_sgd, tpr_sgd, label='SGD')
plot_roc_curve(fpr_logistic, tpr_logistic, label='Logistic')
plot_roc_curve(fpr_svc, tpr_svc, label='SVC')

plt.legend(loc="lower right")
plt.show()

print('AUC_SCORE Random Forest :',roc_auc_score(y_train, y_scores_forest))  # to get the roc_auc score for Random Forest classifier
print('AUC_SCORE SGD :',roc_auc_score(y_train, y_scores_sgd))  # to get the roc_auc score for SGD classifier
print('AUC_SCORE Logistic :',roc_auc_score(y_train, y_scores_logistic))  # to get the roc_auc score for Logistic classifier
print('AUC_SCORE SVC :',roc_auc_score(y_train, y_scores_svc))  # to get the roc_auc score for SVC classifier

################################################### PCA    #########################################################
noise_data = np.random.normal(images,4)  #to add noise to a data

##to remove noise from a data using pca inverse transform
from sklearn.decomposition import PCA
pca = PCA(n_components=0.5)
X2D = pca.fit_transform(noise_data)
filtered = pca.inverse_transform(X2D)  #clean data

##PCA to visulaize data, if the data is not jampacked, then you can use this directly with kmeans clustering
from sklearn.decomposition import PCA
pca = PCA(n_components=2/0.95)
X2D = pca.fit_transform(X)

## Using TSNE when the PCA visualization are highly clustered(jampacked) together i.e, TSNE pushes it further apart
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X2D = tsne.fit_transform(X)

########################################### 
#you can then perform kmeans clustering on the TSNE data/PCA data to further improve the performance

kmeans = KMeans(n_clusters=i, random_state=10)
y_pred = kmeans.fit_predict(X2D)

from scipy.stats import mode
labels = np.zeros_like(y_pred)
for i in range(10):
    mask = (y_pred == i)
    labels[mask] = mode(y[mask])[0]
    
from sklearn.metrics import accuracy_score
accuracy_score(y, labels)

################################################### CLUSTERING  ############################################################
## Using the elbow method to determine the number of clusters
from sklearn.cluster import KMeans

wcss=[]
for i in range(2,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(2,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# i refers to the value of the last abrupt change in the graph
kmeans = KMeans(n_clusters=i, random_state=10)
y_pred = kmeans.fit_predict(X)

#Note: the y_pred is not in correct order so we have to mask it to the true label y and it stored in the variable 'labels'
#To mask our predicted clustered label y_pred to our actual label y
from scipy.stats import mode
labels = np.zeros_like(y_pred)
for i in range(10):
    mask = (y_pred == i)
    labels[mask] = mode(y[mask])[0]
    
from sklearn.metrics import accuracy_score
accuracy_score(y, labels)

## Use DBSCAN to confirm your number of clusters interms of noise
from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
y_pred = dbscan.fit_predict(X)

## Using Silhouette score to validate the number of clusters
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

for n_clusters in range(2,30):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    
X['cluster'] = X['cluster'].astype('category')  #to make the cluster a category feature


#### Gaussian mixture model is used for circular and non-circular data(ellipse, oblique dataset). It uses a probabilistic approach
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=i)
y_gmm = gmm.fit_predict(X)

###################################    Kmeans as Image Color Compression  ################################################## 
#if we want to compress a 1million color image into 16 colors
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=16)
y_pred = kmeans.fit_predict(image_1millioncolors)

image_16colors = kmeans.cluster_centers_[y_pred]

######################### Simple Neural Network  #####################################
#if the batch_size is 4, and we have 4000 dataset size, it will train 1000 data (4000/4) in each epoch that is given randomly
np.random.shuffle(df.values)  #to shuffle data frame before training

from tensorflow.keras import callbacks

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, 
    patience=20,
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Dense(512,activation='relu', input_shape=X_train.shape[1:]),
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
)

history = model.fit(train_processed,y,batch_size=4,epochs=1000,callbacks=[early_stopping], validation_split=0.2)

pd.DataFrame(history.history).plot()
model.evaluate(X_test,y_test)
y_pred = model.predict(X_test)


######################### Simple Neural Network for Image Classification Tasks using CNN #####################################

model = keras.Sequential([
    layers.Conv2D(32,3, activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPool2D(),
    layers.Conv2D(64,3, activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(256,activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10,activation='softmax')
])


#### for binary classification ###
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)


#### for categorical classes  ###
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(X_train,y_train,batch_size=32,epochs=20,verbose=True)


pd.DataFrame(history.history).plot()
model.evaluate(X_test,y_test)
y_pred = model.predict(X_test)


###########################   CONVOLUTION NUERAL NETWORK FOR IMAGES IN FILES   ##############################################
train_dir = 'folder_name/train'
val_dir =  'folder_name/val'
test_dir = 'folder_name/test'

img_height = 128
img_width = 128
batch_size = 32


from keras.preprocessing import image_dataset_from_directory

train_ds = image_dataset_from_directory(train_dir, 
                                        color_mode='grayscale', 
                                        image_size= (img_height,img_width), 
                                        batch_size=batch_size)

val_ds = image_dataset_from_directory(val_dir, 
                                        color_mode='grayscale', 
                                        image_size= (img_height,img_width), 
                                        batch_size=batch_size)

test_ds = image_dataset_from_directory(test_dir, 
                                        color_mode='grayscale', 
                                        image_size= (img_height,img_width), 
                                        batch_size=batch_size)


# training_ds = image_dataset_from_directory(train_dir, 
#                                         validation_split=0.2,
#                                         subset='training',
#                                         seed=42,
#                                         color_mode='grayscale', 
#                                         image_size= (img_height,img_width), 
#                                         batch_size=batch_size)


# validation_ds = image_dataset_from_directory(train_dir, 
#                                         validation_split=0.2,
#                                         subset='validation',
#                                         seed=42,
#                                         color_mode='grayscale', 
#                                         image_size= (img_height,img_width), 
#                                         batch_size=batch_size)



#to visulaize the images
plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis('off')
        

#to augument the datasets       
data_augumentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal'),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.2)
    
])
augumented_train_ds = train_ds.map(lambda x,y:(data_augumentation(x, training=True),y))

#to visualize the augumented datasets
plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1):
    for i in range(9):
        augumented_images = data_augumentation(images)
        ax = plt.subplot(3,3,i+1)
        plt.imshow(augumented_images[i].numpy().astype('uint8'))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis('off')
        
       
 #to process the data for training
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# augumented_train_ds = augumented_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


#training of data
from tensorflow.keras import callbacks

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, 
    patience=20,
    restore_best_weights=True,
)


model = keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1/255),
    layers.Conv2D(32,3, activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64,3, activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(256,activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(train_ds, validation_data=val_ds, epochs=1000, callbacks=[early_stopping])



######################################             PRETRAINED MODEL             #####################################################
base_model = tf.keras.applications.InceptionResNetV2(
                     include_top=False,
                     weights='imagenet',
                     input_shape=(img_height,img_width,3)
                     )

base_model.trainable=False


model = keras.Sequential([ 
        base_model,   
        layers.BatchNormalization(renorm=True),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


###########################################  NLP Concept  #######################################################
for i in df.columns:    ## to replace non-alpha character in a feature with an empty string in a dataframe
    df[i] = df[i].replace("[^a-zA-Z]", " ", regex=True)

    
for i in df.columns:    ## to convert all features in to lower case
    df[i] = df[i].str.lower()

    
rows = []  # to merge all text features into a single row
for i in range(len(df.index)):
    rows.append(' '.join([str(i) for i in df.iloc[i, :]]))
    
   

 df['title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)  #to extract title from a name or analyze a name feature

df['feature'] = df['feature'].replace('character','',regex=True) ## to replace a character with an empty string, if there is a symbol like
                                                                 ##   (+,$), prefix it with (\+,\$) */
    

def get_kmers(sequence, size=6):  #the Kmer counting for DNA sequencing
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]




##########################         Basic Preprocessing of text for NLP classification             ###################################
import nltk
from nltk.stem import PorterStemmer,LancasterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
import re
corpus = []

new_stopwords = set(stopwords.words('english'))
new_stopwords.remove('not')

lemmatizer = WordNetLemmatizer()

for i in range(len(sentences)):
    review = re.sub('<[^<>]+>', ' ', sentences[i])  #for html tags
    review = re.sub('[0-9]+', 'number ', review) #for numbers
    review = re.sub('(http|https)://[^\s]*', 'httpaddr', review) #for urls
    review = re.sub('[^\s]+@[^\s]+', 'emailaddr', review) #for email address
    review = re.sub('[$]+', 'dollar', review) #for dollar signs
    review = re.sub('[^a-zA-Z0-9\s]', '', review)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in new_stopwords]
    review = ' '.join(review)
    corpus.append(review)
    
    
    
    
##########             Advanced Preprocessing of text for NLP classification(Semantic meaning and POS tagging)            ###########

# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
wordnet_lemmatizer = WordNetLemmatizer()

# df['reviews'] ... this is the text you are working on to get your final Lemma

def clean(text):
    text = re.sub('<[^<>]+>', ' ', str(text))  #for html tags
    text = re.sub('[0-9]+', 'number ', str(text)) #for numbers
    text = re.sub('(http|https)://[^\s]*', 'httpaddr', str(text)) #for urls
    text = re.sub('[^\s]+@[^\s]+', 'emailaddr', str(text)) #for email address
    text = re.sub('[$]+', 'dollar', str(text)) #for dollar signs
    text = re.sub('[^a-zA-Z0-9\s]', '', str(text))
    return text

df['Cleaned Reviews'] = df['reviews'].apply(clean)

# POS tagging semantic meaning... nouns(n), verbs(v), adjective(a), adverbs(r)
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}

new_stopwords = set(stopwords.words('english'))
new_stopwords.remove('not')

def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in new_stopwords:
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist 

df['POS tagged'] = df['Cleaned Reviews'].apply(token_stop_pos)

def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

df['Lemma'] = df['POS tagged'].apply(lemmatize)

df[['reviews','Lemma']]


####################### Cosine Similarity of sentences ###################################################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

vec = TfidfVectorizer()

def get_similarities(talk_content, data=df):

    talk_array1 = vec.fit_transform(talk_content).toarray()
 
    sim = []
    pea = []
    for idx, row in data.iterrows():
        details = row['Lemma']
        
        talk_array2 = vec.transform(
            data[data['Lemma'] == details]['Lemma']).toarray()

        cos_sim = cosine_similarity(talk_array1, talk_array2)[0][0]
 
        pea_sim = pearsonr(talk_array1.squeeze(), talk_array2.squeeze())[0]
 
        sim.append(cos_sim)
        pea.append(pea_sim)
 
    return sim, pea


def recommend_talks(talk_content, data=df):
 
    data['cos_sim'], data['pea_sim'] = get_similarities(talk_content)
 
    data.sort_values(by=['cos_sim', 'pea_sim'], ascending=[
                     False, False], inplace=True)
 
    return data[['reviews', 'Lemma']].head()  #it returns the most common text 


new_df = recommend_talks(['Any similar text that you want to check'])
new_df

#############################         To generate SENTIMENT ANALYSIS using VADER on the Lemma(cleaned text)     ###########################
# !pip install vaderSentiment
#compound ranges from -1 to 1
#positive ranges from 0.5 to 1
#neutral ranges from 0 to 0.499
#negative ranges from  less than 0 to -1

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


def vadersentimentanalysis(review):
    vs = analyzer.polarity_scores(review)
    return vs['compound']

df['Sentiment'] = df['Lemma'].apply(vadersentimentanalysis)  #using the feature 'lemma' from the cleaned text

def vader_analysis(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound < 0 :
        return 'Negative'
    else:
        return 'Neutral'
df['Analysis'] = df['Sentiment'].apply(vader_analysis)
df.head()


#################           To generate WORD2CLOUD on the text        ##############################################
#word2cloud is used to get the keywords used in a text
# !pip install wordcloud

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df['text'])  #using the cleaned preprocessed text
    
######################### To split a corpus into categories of various fileids i.e X and y data  #############################
documents = []
for category in corpus.categories():
    for fileid in corpus.fileids(category):
        text = corpus.raw(fileid)
        documents.append((text, category))
        
np.random.shuffle(documents)
data = np.array(documents)

#############################   Full Pipleline of using Naive Bayes Model to train a corpus in NLTK #####################################
documents = []
for category in corpus.categories():
    for fileid in corpus.fileids(category):
        text = corpus.words(fileid)
        documents.append((text, category))
        
np.random.shuffle(documents)


all_words = []

for word in corpus.words():
    all_words.append(word)

all_words = nltk.FreqDist(all_words)

'''
all_words = []

for row in corpus:
    words = word_tokenize(row)
    for word in words:
        all_words.append(word)
        
'''

vocabulary = list(all_words.keys())[:2000] #to select the most common 2000 words

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def find_features(document):
    words = set(document)
    words =  [lemmatizer.lemmatize(word) for word in words]
    features = {}
    for w in vocabulary:
        features[w] = (w in words)
    return features


data = [(find_features(rev), category) for (rev, category) in documents]
data = np.array(data)

#spliting of dataset...sets of dictionary X and target y
train_set = data[:1900] 
test_set = data[1900:]

classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.classify(X) 
nltk.classify.accuracy(classifier, test_set)  #to check accuracy score

classifier.show_most_informative_features(15)  #to check the 15 most important features


###############################   SklearnClassifier as a wrapper on sklearn classifiers  ############################
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

classifier = SklearnClassifier(MultinomialNB())  #the SklearnClassifier serves as a wrapper
classifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


##################################    Chunking, Chinking , Named Entity Recognition #############################################
# chunking... to group words
for sent in test_tokenized:
    words = word_tokenize(sent)
    tagged = nltk.pos_tag(words)
    chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP><NN>?} """
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    chunked.draw()


# chunking and chinking .... chinking means to remove from a group
for sent in test_tokenized:
    words = word_tokenize(sent)
    tagged = nltk.pos_tag(words)
    chunkGram = r"""Chunk: {<.*>+}  
                                }<VB.?|IN|DT>+{"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    chunked.draw()

    
# Named Entity Recognition
for sent in test_tokenized:
    words = word_tokenize(sent)
    tagged = nltk.pos_tag(words)
    namedEnt = nltk.ne_chunk(tagged, binary=True)
    namedEnt.draw()

    
#####################################           LSTM/RNN/Bi-Directional FULL PIPELINE       ##########################################
#After cleaning the corpus and stemming it

import tensorflow
from tensorflow import keras
from keras import layers
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

vocabulary_size = 5000

onehot_repr = [one_hot(words,vocabulary_size) for words in corpus]
onehot_repr

# pd.Series(corpus).apply(lambda x: len(x.split(' '))).max() #to get the sentence length

sent_length = 50
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length) #you can also use post padding
embedded_docs    

np.argmax(embedded_docs[:,0]) #to hypertune the required number of sentence length...check if it gives 0

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(embedded_docs,y,test_size=0.33)

embedding_vector_features = 40  #each word will have 40 features like the google-300-features

model = keras.Sequential([
    layers.Embedding(vocabulary_size, embedding_vector_features, input_length=sent_length),
    layers.LSTM(100),
#     layers.Bidirectional(layers.LSTM(100)),  #bi-directional 
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
  

############################################        NLP BOOK TRICKS     ###################################################################
all_words = []
for sent in df.feature.str.split(' '):
    for word in sent:
        if word == '':
            continue
        else:
            all_words.append(word)
            
text = nltk.Text(all_words)            

text.concordance('word') #to check the occurence of a word in a text
text.similar('word')  #to check the contextual meaning of a word
text.generate(100)  #to generate 100 random text
text.dispersion_plot(['word1','word2'])  #to graph the instance of the occurrence of a word in a text
text.collocations()  #to get collocations

