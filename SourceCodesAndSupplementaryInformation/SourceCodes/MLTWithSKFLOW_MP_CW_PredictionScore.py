import numpy as np
import sys
import warnings
from scipy import interp
import pylab as pl
import pandas as pd
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from sklearn import metrics
import multiprocessing
from sklearn.utils import class_weight
# from sklearn.externals import joblib
# import pickle
# import dill

# Not show unnecessary Warning filter
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ResourceWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)



class Execute_machine_learning():

    def data_generation(self,csv_path):
        data_table = pd.read_csv(csv_path)

        last_index_column = len(data_table.columns)

        pred_data = data_table[data_table.columns[1: last_index_column]].values
        outcome_data = data_table[data_table.columns[0]].values

        print("Inside data generator........")

        return pred_data, outcome_data

    def parameter_tuning_SVM(self, X, y, nfolds, cls_wgt):
        parameter_svm = [
            # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            # {'C': [1, 10, 100, 1000], 'kernel': ['poly']},
            {'C': [1, 10, 100, 1000], 'gamma': ["auto", 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
            # {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid']}
        ]

        svm_grid_search = GridSearchCV(estimator=SVC(probability=True, random_state=42, class_weight=cls_wgt),
                                       cv=nfolds, param_grid=parameter_svm, n_jobs=-1)

        svm_grid_search.fit(X, y)

        # print("Best score achieved:", svm_grid_search.best_score_)
        # print("Best Parameters:", svm_grid_search.best_params_)
        # print("Scores:", svm_grid_search.best_estimator_)

        fileforSVM = open("SVM_BestParameterAndAccuracy.txt", "w+")

        fileforSVM.write("Best accuracy score achieved = {} \n".format(svm_grid_search.best_score_))
        fileforSVM.write("Best Parameters: {} ".format(svm_grid_search.best_params_))
        fileforSVM.close()

        print("Inside SVM parameter tuning........")

        return svm_grid_search.best_estimator_

    def parameter_tuning_RF(self, X, y, nfolds, cls_wgt):
        parameter_rf = {'bootstrap': [True, False],
                        'max_depth': [None, 90, 100],
                        'max_features': ["auto", 2, 3],
                        'min_samples_leaf': [1, 2, 3],
                        'min_samples_split': [2, 4, 8],
                        'n_estimators': [10, 100, 200, 300, 500]}

        rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42, class_weight=cls_wgt),
                                      cv=nfolds, param_grid=parameter_rf, n_jobs=-1)

        rf_grid_search.fit(X, y, class_weight=cls_wgt)

        fileforRF = open("RF_BestParameterAndAccuracy.txt", "w+")

        fileforRF.write("Best accuracy score achieved = {} \n".format(rf_grid_search.best_score_))
        fileforRF.write("Best Parameters: {} ".format(rf_grid_search.best_params_))
        fileforRF.close()

        print("Inside RF parameter tuning........")

        return rf_grid_search.best_estimator_

    def parameter_tuning_DNN(self, X, y, nfolds):
        parameter_dnn = {"hidden_units": [[10, 20, 10], [20, 40, 20], [50, 100, 50], [10, 20, 20, 10]],
                         "learning_rate": [0.1, 0.2],
                         "batch_size": [10, 32]}

        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X)

        classifier_DNN = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                        hidden_units=[10, 20, 10], n_classes=2,
                                                        config=tf.contrib.learn.RunConfig(tf_random_seed=232))

        dnn_grid_search = GridSearchCV(classifier_DNN, cv=nfolds, scoring="accuracy",
                                       param_grid=parameter_dnn, fit_params={'steps': [200, 400]})

        model = dnn_grid_search.fit(X, y)

        print("DNN grid search:", list(model))


        return dnn_grid_search.best_estimator_

    def performance_measure(self, X, y, model, nfolds, modelIndex, id_list, input_fileName):

        sen_list = []
        spec_list = []
        acc_list = []
        pre_list = []
        mcc_list = []
        f1_list = []

        tpr_list = []
        mean_fpr = np.linspace(0, 1, 100)
        # auc_list = []

        model_name = ("SVM", "RF", "DNN")
        # this list is used for store identifier with prediction score
        list_Model_Identifier_Prediction = []

        skf = StratifiedKFold(n_splits = nfolds, random_state=423)

        for train_index, test_index in skf.split(X,y):
            # print("Train:", train_index, "Test:", test_index)
            if modelIndex == 2:
                # oSaver = tf.train.Saver(model)
                # oSess = tf.Session()
                probability_model = model.fit(X[train_index], y[train_index], steps=2000).predict_proba(
                    X[test_index], as_iterable=False)
                prediction_model = model.fit(X[train_index], y[train_index], steps=2000).predict(
                    X[test_index], as_iterable=False)
                # model.fit(X[train_index], y[train_index])

                fpr, tpr, thresholds = metrics.roc_curve(y[test_index], probability_model[:, 1])

                # list_Model_Identifier_Prediction.append([model_name[modelIndex], list(np.array(id_list)[test_index]), probability_model])
                list_Model_Identifier_Prediction.append([list(np.array(id_list)[test_index]), list(probability_model[:, 1]), list(probability_model[:, 0])])
                # print(list_Model_Identifier_Prediction)
                tpr_list.append(interp(mean_fpr, fpr, tpr))
                tpr_list[-1][0] = 0.0

                conf_matrix = metrics.confusion_matrix(y[test_index], prediction_model)

            else:
                probability_model = model.fit(X[train_index], y[train_index]).predict_proba(X[test_index])
                prediction_model = model.fit(X[train_index], y[train_index]).predict(X[test_index])
                # model.fit(X[train_index], y[train_index])


                fpr, tpr, thresholds = metrics.roc_curve(y[test_index], probability_model[:, 1])

                # list_Model_Identifier_Prediction.append([model_name[modelIndex], list(np.array(id_list)[test_index]), probability_model])
                list_Model_Identifier_Prediction.append([list(np.array(id_list)[test_index]), list(probability_model[:, 1]), list(probability_model[:, 0])])
                # print(list_Model_Identifier_Prediction)

                tpr_list.append(interp(mean_fpr, fpr, tpr))
                tpr_list[-1][0] = 0.0

                conf_matrix = metrics.confusion_matrix(y[test_index], prediction_model)
                # print(conf_matrix)

            new_list_CM = []

            for i in conf_matrix:
                for j in i:
                    new_list_CM.append(j)

            TP = float(new_list_CM[0])
            FP = float(new_list_CM[1])
            FN = float(new_list_CM[2])
            TN = float(new_list_CM[3])

            # print("TP:", TP, "FP:", FP, "FN:", FN,"TN:", TN)
            # sensitivity = specificity = accuracy = precision = f1 = 0
            if (TP + FN) > 0:
                sensitivity = round(float(TP / (TP + FN)), 2)
            else:
                sensitivity = 0
            if (TN + FP) > 0:
                specificity = round(float(TN / (TN + FP)), 2)
            else:
                specificity = 0
            if (TP + FP + FN + TN) > 0:
                accuracy = round(float((TP + TN) / (TP + FP + FN + TN)), 2)
            else:
                accuracy = 0
            if (TP + FP) > 0:
                precision = round(float(TP / (TP + FP)), 2)
            else:
                precision = 0
            try:
                mcc = round(metrics.matthews_corrcoef(y[test_index], prediction_model), 2)
            except:
                print("Error in mcc")
                pass
            if (sensitivity + precision) > 0:
                # f1 = round(metrics.f1_score(y[test_index], prediction_model), 2)
                f1 = 2 * ((sensitivity * precision)/(sensitivity + precision))
            else:
                f1 = 0

            # store the value in list of performance measure
            sen_list.append(sensitivity)
            spec_list.append(specificity)
            acc_list.append(accuracy)
            pre_list.append(precision)
            mcc_list.append(mcc)
            f1_list.append(f1)


        sen_mean = round(float(sum(sen_list))/float(len(sen_list)),3)
        spec_mean = round(float(sum(spec_list))/float(len(spec_list)),3)
        acc_mean = round(float(sum(acc_list))/float(len(acc_list)),3)
        pre_mean = round(float(sum(pre_list))/float(len(pre_list)),3)
        mcc_mean = round(float(sum(mcc_list))/float(len(mcc_list)),3)
        f1_mean = round(float(sum(f1_list))/float(len(f1_list)),3)


        mean_tpr = np.mean(tpr_list, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr,mean_tpr)


        perf_header = ("sensitivity", "specificity", "accuracy", "precision", "mcc", "f1", "auc")
        perf_value = (sen_mean, spec_mean, acc_mean, pre_mean, mcc_mean, f1_mean, round(mean_auc,3))
        # print("Header:",perf_header, "Value:", perf_value)

        #  This section will write the prediction score for each instance in csv file
        id_with_Predction_Score = pd.DataFrame(data=list_Model_Identifier_Prediction, columns=["Identifier", "Positive_probability", "Negative_probability"])
        # id_with_Predction_Score.to_csv(model_name[modelIndex] + "_" + str(id_fileName), index=False)
        np_Identifier = np.concatenate(id_with_Predction_Score["Identifier"].values)
        np_Positive_probability= np.concatenate(id_with_Predction_Score["Positive_probability"].values)
        np_Negative_probability = np.concatenate(id_with_Predction_Score["Negative_probability"].values)
        final_id_with_Predction_Score = pd.DataFrame({'Identifier':np_Identifier, 'Probability_BacteriaTargeted':np_Positive_probability, 'Probability_VirusTargeted':np_Negative_probability})
        final_id_with_Predction_Score.to_csv("PredictionScore_" + model_name[modelIndex] + "_" + str(input_fileName), index=False)

        return perf_header, perf_value, mean_tpr, mean_fpr


    def main_program(self, input_file, no_fold, input_identifier_file):

        pred_var, outcome_var = Execute_machine_learning().data_generation(input_file)

        read_identifier_list = pd.read_csv(input_identifier_file)

        identifier_list = list(read_identifier_list[read_identifier_list.columns[0]])


        class_weight_values = class_weight.compute_class_weight('balanced', np.unique(outcome_var), outcome_var)
        # data into dictionary format..
        class_weights = dict(zip(np.unique(outcome_var), class_weight_values))

        perfromance_values = []

        tpr_list = []
        fpr_list = []


        SVM = Execute_machine_learning().parameter_tuning_SVM(pred_var, outcome_var, no_fold, class_weights)
        RF = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weights)

        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(pred_var)

        DNN = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10],
                                            n_classes=2, config= tf.contrib.learn.RunConfig(num_cores=20, tf_random_seed=42))

        classifiers = (SVM, RF, DNN)

        for i, classifier in enumerate(classifiers):
            performance_header, performance_value, tpr, fpr = Execute_machine_learning().\
                performance_measure(pred_var, outcome_var, classifier, no_fold, i, identifier_list, input_file)
            perfromance_values.append(performance_value)
            tpr_list.append(tpr)
            fpr_list.append(fpr)


        performance_result = pd.DataFrame(perfromance_values, index=("SVM", "RF", "DNN"), columns=performance_header)

        performance_result.plot.bar()
        pl.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol = 4)
        pl.savefig("CW_Performance_summary_" + str(no_fold) + "foldCV_" + str(input_file).replace(".csv", ".png"))
        # pl.savefig("TestPerforamnec_SKflow_summary.png")
        pl.close()

        performance_result.to_csv("CW_Performance_summary_" + str(no_fold) + "foldCV_" + str(input_file))
        # performance_result.to_csv("TestPerforamnec_SKflow.csv")

        model_short_name = ("SVM", "RF", "DNN")

        pl.plot([0, 1], [0, 1], '--', lw=2, label="Random(AUC = 0.5)")

        for i, tpr_value in enumerate(tpr_list):
            mean_auc = metrics.auc(fpr_list[i], tpr_value)
            pl.plot(fpr_list[i], tpr_value, '-', label=model_short_name[i] + '(AUC = %0.2f)' % mean_auc, lw=2)

        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC Curve')
        pl.legend(loc="lower right")
        pl.savefig("CW_CombineRocCurve_" + str(input_file).replace(".csv", ".png"))
        # pl.savefig("Test_SKflowRocCurve.png")
        pl.close()


if __name__=="__main__":
    import argparse

    multiprocessing.set_start_method('forkserver', force=True)
    parser = argparse.ArgumentParser()

    # get arguments from command line
    parser.add_argument("-f", "--filepath",
                        required=True,
                        default=None,
                        help="Path to target CSV file")
    parser.add_argument("-n", "--n_folds",
                        required=None,
                        default=5,
                        help="n_folds for Cross Validation")
    parser.add_argument("-u", "--identifier_file",
                        required=None,
                        default=None,
                        help="Path to target CSV file")

    # parse the argument from command line
    args = parser.parse_args()
    # Create instance of class
    objEML = Execute_machine_learning()
    # pass the command line argument to the method
    objEML.main_program(args.filepath, int(args.n_folds), args.identifier_file)


    exit()
