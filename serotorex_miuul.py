import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np
import matplotlib.pyplot as plt
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
#cat ve num kolonların çıktı kırılımında ortalamaları
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

#########............##########..............##########

df_ = pd.read_csv(r"D:\masaüstü\MİUUL BOOTCAMP\DRUG_ML_PROJE\drug200.csv")
df = df_.copy()

check_df(df)

#değişkenlerin türleri ayrıştırılır

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

#değişkenler incelenir : cat,num
for col in cat_cols:
    cat_summary(df, col)

# Veri setindeki "Drug" değişkeninin dağılımını elde etmek için değerleri sayıyoruz
drug_counts = df['Drug'].value_counts()
# Seaborn ile pasta dilimi grafiği oluşturuyoruz
plt.figure(figsize=(10, 5))
sns.set(style="whitegrid")
plt.pie(drug_counts, labels=drug_counts.index, autopct='%1.1f%%')
# Grafiği gösteriyoruz
plt.show()
plt.show(block=True)

#İlaçların kan basıncındaki dağılım grafiği
plt.figure(figsize=(10, 5))
sns.countplot(x="BP", data=df, hue="Drug", palette="Spectral")
plt.title("Distribution of Blood Pressure by Drug")
plt.xlabel("Blood Pressure")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Drug")
plt.show()
plt.show(block=True)

#İlaçların Na_to_K dağılım grafiği
# Bar plot oluşturma
plt.figure(figsize=(10, 5))
sns.barplot(x="Drug", y="Na_to_K", data=df, palette="Spectral")
# Eksen etiketlerini düzenleme
plt.xlabel("Drug")
plt.ylabel("Na_to_K")
# Başlığı ekleme
plt.title("Sodium-Potassium Ratio (Na_to_K) for Each Drug")
# Grafikleri gösterme
plt.show()
plt.show(block=True)

#İlaçların kolesterol dağılım grafiği
plt.figure(figsize=(10, 5))
sns.countplot(x="Cholesterol", data=df, hue="Drug", palette="Spectral")
plt.title("Distribution of Cholesterol by Drug")
plt.xlabel("Cholesterol")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Drug")
plt.show()
plt.show(block=True)

#İlaçların belirlenen aralıktaki yaşlara göre dağılım grafiği
df.loc[df['Age'] < 35, "NEW_AGE_CAT"] = 'young'
df.loc[(df['Age'] >= 35) & (df['Age'] <= 55), "NEW_AGE_CAT"] = 'middleage'
df.loc[df['Age'] > 55, "NEW_AGE_CAT"] = 'old'
df.head()
sns.countplot(x="NEW_AGE_CAT", hue="Drug", data=df)
plt.xlabel("NEW_AGE_CAT")
plt.ylabel("Count")
plt.title("Drug by NEW_AGE_CAT")
plt.show()
plt.show(block=True)

#kolesterol,kan basıncı,na_to_k 'nın yaş ve cinsiyete göre dağılımı
age_categories = ["young", "middleage", "old"]
bp_values = ["HIGH", "LOW", "NORMAL"]
cholesterol_values = ["HIGH", "NORMAL"]
na_to_k_values = df["Na_to_K"].unique()
fig, axes = plt.subplots(len(age_categories), len(bp_values), figsize=(12, 12))
for i, age_category in enumerate(age_categories):
    for j, bp in enumerate(bp_values):
        for k, cholesterol in enumerate(cholesterol_values):
            subset = df[(df["NEW_AGE_CAT"] == age_category) & (df["BP"] == bp) & (df["Cholesterol"] == cholesterol)]
            ax = axes[i][j]
            sns.countplot(x="Sex", data=subset, ax=ax)
            ax.set_title(f"Age Category: {age_category}, BP: {bp}, Cholesterol: {cholesterol}" , fontsize=8)
            ax.set_xlabel("Sex")
            ax.set_ylabel("Count")
plt.tight_layout()
plt.show()
plt.show(block=True)

df[num_cols].describe().T

######################################################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

#na_to_k değişkeni için aykırı değer kontrolü yapıldı
low,up =outlier_thresholds(df,"Na_to_K",q1=0.25, q3=0.75)
#aykırı değerleri görüntüledik
df[(df["Na_to_K"] < low) | (df["Na_to_K"] > up)].head()
df[(df["Na_to_K"] < low) | (df["Na_to_K"] > up)].index
check_outlier(df,"Na_to_K",q1=0.25, q3=0.75)
sns.boxplot(x=df["Na_to_K"])
plt.show()
plt.show(block=True)

#Encoding
#sınıf sayısı 2 olup,int veya float olmayan değişkenleri binary col olarak adlandırdık
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
#bu değişkenleri label encoderdan geçirdik,böylece kategorik tipte olan değişkeni 1 ,0 şeklinde ifade ettik
for col in binary_cols:
    label_encoder(df, col)
df.head()

#Ordinal olan değişkenleri label encoderdan geçirdik ve iki yeni değişken oluşturduk
label_encoder = LabelEncoder()
df["BP_encoded"] = label_encoder.fit_transform(df["BP"])
df["Drug_encoded"] = label_encoder.fit_transform(df["Drug"])

df["BP_encoded"].nunique()
df["Drug_encoded"].nunique()

df = df.drop(['BP','Drug'], axis = 1) #sayısalları bıraktım
df.head()

# StandardScaler
ss = StandardScaler()
df["Na_to_K_standard_scaler"] = ss.fit_transform(df[["Na_to_K"]])
df.head()

#korelasyon matrisi
korelasyon_matrisi = df.corr()
print(korelasyon_matrisi)

#Feature extraction
from statsmodels.stats.proportion import proportions_ztest
df.loc[df['Age'] < 35, "NEW_AGE_CAT"] = 'young'
df.loc[(df['Age'] >= 35) & (df['Age'] <= 55), "NEW_AGE_CAT"] = 'middleage'
df.loc[df['Age'] > 55, "NEW_AGE_CAT"] = 'old'
df["NEW_AGE_CAT"]

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_AGE_CAT"] == "old", "Cholesterol"].sum(),
                                             df.loc[df["NEW_AGE_CAT"] == 'middleage', "Cholesterol"].sum()],


                                      nobs=[df.loc[df["NEW_AGE_CAT"] == "old", "Cholesterol"].shape[0],
                                            df.loc[df["NEW_AGE_CAT"] == 'middleage', "Cholesterol"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

table = df.groupby(['NEW_AGE_CAT', 'Drug_encoded']).size().unstack()

# Tabloyu görüntüleme
print(table)
table.to_csv('tablo.csv', index=True)
# Tabloyu açma
table = pd.read_csv('tablo.csv', index_col=0)
# Tabloyu görüntüleme
print(table)
#p value =0.09 olduğu için p hipotezi kabul edilir.İki değişken arasında anlamlı bir farklılık yoktur diyebiliriz.

####################################################
# Özellikler ve hedef değişken arasında ayrım yapma
X = df.drop(['Drug_encoded',"NEW_AGE_CAT"] ,axis=1)
y = df['Drug_encoded']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
X_train.shape, X_test.shape
X_train.dtypes
X_train.head()

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)

from sklearn.metrics import accuracy_score
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
y_pred_train_gini = clf_gini.predict(X_train)
y_pred_train_gini
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))

from sklearn.metrics import f1_score
# Test veri seti için F1 skoru hesaplama
f1 = f1_score(y_test, y_pred_gini, average='weighted')
print('F1 score: {0:0.4f}'.format(f1))

from sklearn.metrics import recall_score
# Test veri seti için recall (duyarlılık) hesaplama
recall = recall_score(y_test, y_pred_gini, average='weighted')
print('Recall score: {0:0.4f}'.format(recall))

from sklearn import tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(X_train, y_train))
plt.show()
plt.show(block=True)

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
print('F1 score: {0:0.4f}'.format(f1))
print('Recall score: {0:0.4f}'.format(recall))

