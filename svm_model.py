from sklearn.svm import SVC
# Create a classifier: a support vector classifier
#svm_model = SVC(kernel='poly', C =0.8) # will experiment with different 
#hyperparameters obtained from randomized search
#svm_model = SVC(kernel = 'poly', C= 1, degree =2)
#svm_model = SVC(kernel = 'poly', C= 0.8, degree =2)
#svm_model = SVC(kernel = 'poly',C =0.8)
# svm_model = SVC(kernel = 'poly',C =3.363585661014858, gamma =0.25 ,degree=3 )
#svm_model = SVC(kernel = 'rbf',C =16, gamma =0.25 ,degree=3 )--used in balanced data
svm_model = SVC(kernel = 'poly',C =1.0, gamma =0.25 ,degree=3, class_weight='balanced' )
#svm_model = SVC(kernel = 'poly',C =8.0, gamma =0.25 ,degree=4 )
#svm_model = SVC(kernel = 'poly',C =2.378414230005442, gamma =0.25 ,degree=3)

