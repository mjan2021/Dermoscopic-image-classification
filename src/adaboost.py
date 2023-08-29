from imports import *
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def read_file(path):
    meta = pd.read_csv(path, index_col=False)
    return meta


print(f'main(): Loading Data..')
dataframe = read_file('../data/Augmented/flip.csv')
y = dataframe['dx']
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
train = train_datagen.flow_from_dataframe(dataframe, x_col='img_path', y_col='dx', target_size=(32, 32), batch_size=8, class_mode='categorical', shuffle=True)

print(f'main(): Setting Up Model parameters..')
base_classifier = DecisionTreeClassifier(max_depth=1)
n_estimators = 3  # You can adjust this as needed
adaboost_classifier = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=n_estimators)


print(f'main(): Training Started..')
adaboost_classifier.fit(train, y)