import os
import numpy as np
from joblib import dump, load
from time import strftime, gmtime

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


class ClassifierManager(object):
    def __init__(self, features_file, classifier_path, verbose_level):
        self._verbose_level = verbose_level
        self._features_file = features_file
        self._classifier_path = classifier_path

    def _open_files(self):
        x_train = np.loadtxt(self._features_file, dtype=np.float)
        y_train = np.loadtxt(self._labels_file, dtype=np.float)
        if self._verbose_level > 0:
            print("Features shape: {}\nLabels shape: {}".format(x_train.shape, y_train.shape))
        if len(x_train.shape)==0 | len(y_train.shape)==0 | x_train.shape[0]!=y_train.shape[0]:
            raise IOError('Can\'t use files to fit the classifier')
        if x_train.shape[0]!=y_train.shape[0]:
                print("Can't fit classifier.\nX_train shape: {} \t Y_train shape: {}"
                      .format(x_train.shape, y_train.shape))
        return x_train, y_train

    def _get_mean_descriptors(self, x_train, y_train):
        if len(x_train.shape)!=0 and len(y_train.shape)!=0 and x_train.shape[0]==y_train.shape[0]:
            ulabels = np.sort(np.unique(y_train))
            y = np.arange(0, len(ulabels))
            x = []
            for i in ulabels:
                xx = x_train[y_train==i]
                xx /= np.sqrt(np.sum(xx**2, axis=1, keepdims=True))
                x.append(np.mean(xx, axis=0))
        x = np.asarray(x)
        return x, y

    def train(self):
        x_train, y_train = self._open_files()
        if self._train_with_centroids:
            x_train, y_train = self._get_mean_descriptors(x_train, y_train)
        # save 20% of data for performance evaluation
        X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

        param = [
            {
                "kernel": ["linear"],
                "C": [1, 10, 100, 1000]
            },
            {
                "kernel": ["rbf"],
                "C": [1, 10, 100, 1000],
                "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
            }
        ]

        # request probability estimation
        svm = SVC(probability=True)

        # 10-fold cross validation
        clf = GridSearchCV(svm, param, cv=10, n_jobs=-1, verbose=3)

        clf.fit(X_train, y_train)

        output_path = os.path.join(self._classifier_path, 'model_'+strftime("%Y-%m-%d_%H:%M:%S", gmtime()))
        dump(clf.best_estimator_, output_path)
        print('Classifier saved at: {}'.format(output_path))

        y_predict = clf.predict(X_test)

        labels = sorted(list(set(self.y_train)))

        if self._verbose_level > 0:
            print("="*60)
            print("Best parameters set:")
            print(clf.best_params_)
            print("\nConfusion matrix:")
            print("Labels: {0}\n".format(",".join(labels)))
            print(confusion_matrix(y_test, y_predict, labels=labels))
            print("\nClassification report:")
            print(classification_report(y_test, y_predict))
            print("=" * 60)

    def classify_faces(self, descriptors):
        person_ids = []
        confidences = []
        descriptors = np.asarray(descriptors)
        if descriptors.shape[0] > 1:
            descriptors = descriptors.squeeze()
        else:
            descriptors = descriptors.reshape(1, -1)
        probs = self._classifier.predict_proba(descriptors)
        labels = self._classifier.predict(descriptors)
        person_id = np.argmax(probs, axis=1)
        idx = np.arange(len(person_id))
        confidence = probs[idx, person_id]
        person_ids.append(person_id)
        confidences.append(confidence)
        return person_ids, confidences
