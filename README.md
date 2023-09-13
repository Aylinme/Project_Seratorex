Amaç: Aynı hastalığa sahip gelecekteki hastalar için uygun ilacı tahmin etmeye yönelik bir karar ağacı modeli oluşturmaktır. Veri seti Sex: cinsiyet, BP: kan basıncı, sodyum-potasyum dengesi: Na_to_K ve kolesterol: Cholesterol gibi özelliklerden oluşur ve hedef değişken, her hastanın yanıt verdiği ilaçtır (İlaç A, İlaç B, İlaç C, İlaç X veya İlaç Y). 

Karar ağacı modelini oluşturmak için izlenen adımlar: 

Verileri ön işleme: Bu kısımda eksik değerlerin işlenmesini, kategorik değişkenlerin (Sex, BP, Cholestorol) kodlanmasını ve gerekirse sayısal özelliklerin normalleştirilmesini içerir. 

Veri kümesini bölme: Veri kümesi test ve train olarak ikiye bölünmüştür. Train seti, karar ağacı modelini oluşturmak için, test seti ise performansını değerlendirmek için kullanılmıştır. 

Karar ağacını oluşturma: Karar ağacı modelini oluşturmak için train seti kullanılmıştır. Karar ağacı algoritması, tahminlerde bulunmak için en bilgilendirici özellikleri ve ayrım noktalarını otomatik olarak seçecektir. 

Modeli değerlendirmesi: Karar ağacı modelinin performansını değerlendirmek için test setini kullanılmıştır. Modelin ilaç yanıtını ne kadar iyi tahmin ettiğini değerlendirmek için accuracy, recall ve F1 skorları hesaplanmıştır. 

Yeni hastalar için ilacı tahmini: Karar ağacı modeli eğitilip değerlendirildikten sonra, aynı hastalığa sahip yeni bir hasta için uygun ilacı tahmin etmek için kullanılabilir. Hastanın özelliklerini (yaş, cinsiyet, kan basıncı, sodyum-potasyum ve kolesterol) karar ağacına girdi olarak eklendiğinde tahmin edilen ilaç çıkacaktır. 

#########################################################################################

# Project_Seratorex
Objective: The aim is to create a decision tree model for predicting the appropriate drug for future patients with the same medical condition
The dataset consists of features such as Sex (gender), BP (blood pressure), sodium-potassium balance (Na_to_K), and cholesterol (Cholesterol), with the target variable being the drug to which each patient responds (Drug A, Drug B, Drug C, Drug X, or Drug Y).

Steps followed to create the decision tree model:

Data Preprocessing: This part involves handling missing values, encoding categorical variables (Sex, BP, Cholesterol), and normalizing numerical features if necessary.

Data Splitting: The dataset has been divided into two parts, namely the training set and the test set. The training set is used to build the decision tree model, while the test set is used to evaluate its performance.

Building the Decision Tree: The decision tree model is constructed using the training set. The decision tree algorithm will automatically select the most informative features and decision points to make predictions.

Model Evaluation: The test set is used to assess the performance of the decision tree model. Accuracy, recall, and F1 scores are calculated to evaluate how well the model predicts drug responses.

Predicting the Drug for New Patients: After training and evaluating the decision tree model, it can be used to predict the appropriate drug for a new patient with the same medical condition. When the patient's features (age, gender, blood pressure, sodium-potassium, and cholesterol) are input into the decision tree, the predicted drug will be determined.
