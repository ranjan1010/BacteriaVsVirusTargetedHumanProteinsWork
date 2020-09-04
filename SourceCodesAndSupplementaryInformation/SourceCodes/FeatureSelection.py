import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import class_weight
# import MLTwithPython3 as MLT


class Execute_feature_selection:

    def data_generation(self, csv_path):
        # This is used to get the actual path of csv file
        self.csv_path = csv_path

        data_table = pd.read_csv(self.csv_path)

        last_index_column = len(data_table.columns)

        X = data_table[data_table.columns[1: last_index_column]].values
        y = data_table[data_table.columns[0]].values

        return data_table, X, y

    def generate_dataframe_feature_selection(self, selected_idX, data):

        # Added 1 since X has one less column (Indicator) than data_table
        index_data_table = np.array(selected_idX) + 1

        # add Indicator column header to index_data_table
        index_data_table_full = np.insert(index_data_table, 0, 0)

        selected_features_header = data.columns[index_data_table_full]
        selected_features_data = data[selected_features_header].values

        return pd.DataFrame(selected_features_data, columns=selected_features_header)

    def feature_selection_univariate(self, data_table, X, y, input_file):

        no_best_features = int(len(X[1, :]) / 10)
        score_fun_kbest = [chi2, f_classif, mutual_info_classif]
        score_fun_name = ["chi2", "f_classif", "mutual_info_classif"]

        for i, sfv in enumerate(score_fun_kbest):
            kbest = SelectKBest(score_func=sfv, k=no_best_features).fit(X, y)

            idxs_selected = kbest.get_support(indices=True)

            selected_data_table = Execute_feature_selection().generate_dataframe_feature_selection(idxs_selected, data_table)
            selected_data_table.to_csv("Feature_selection_univariate_"+ score_fun_name[i] + str(input_file), index=False)

    def feature_selection_RFE(self, data_table, X, y, input_file):

        no_best_features = int(len(X[1, :]) / 10)

        # This will call the svc parameter tuning function of MLTwithPython3
        # svc_best_model = MLT.Execute_machine_learning().parameter_tuning_SVM(X, y, 5)
        class_weight_values = class_weight.compute_class_weight('balanced', np.unique(y), y)
        # data into dictionary format..
        class_weights = dict(zip(np.unique(y), class_weight_values))

        RF = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weights)
        recursive_feature_selection = RFE(estimator=RF, n_features_to_select=no_best_features).fit(X,y)


        idxs_selected = recursive_feature_selection.get_support(indices=True)

        selected_data_table = Execute_feature_selection().generate_dataframe_feature_selection(idxs_selected, data_table)
        selected_data_table.to_csv("Feature_selection_RF_RFE_" + str(input_file), index=False)

    def feature_selection_SFM(self, data_table, X, y, input_file):
        lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(X, y)
        model_sfm = SelectFromModel(lsvc, prefit=True)

        idxs_selected = model_sfm.get_support(indices=True)

        selected_data_table = Execute_feature_selection().generate_dataframe_feature_selection(idxs_selected, data_table)
        selected_data_table.to_csv("Feature_selection_LSVC_SFM_"+ str(input_file), index=False)

    def feature_selection_TBFS(self, data_table, X, y, input_file):
        extra_tree_classifier = ExtraTreesClassifier().fit(X, y)
        model_sfm_tbfs = SelectFromModel(extra_tree_classifier, prefit=True)

        idxs_selected = model_sfm_tbfs.get_support(indices=True)

        selected_data_table = Execute_feature_selection().generate_dataframe_feature_selection(idxs_selected, data_table)
        selected_data_table.to_csv("Feature_selection_ETC_SFM_TBFS_" + str(input_file), index=False)


    def main_program(self, input_file):
        self.input_file = input_file

        data, prediction_var, outcome_var = Execute_feature_selection().data_generation(self.input_file)

        Execute_feature_selection().feature_selection_univariate(data, prediction_var, outcome_var, input_file)
        Execute_feature_selection().feature_selection_RFE(data, prediction_var, outcome_var, input_file)
        Execute_feature_selection().feature_selection_SFM(data, prediction_var, outcome_var, input_file)
        Execute_feature_selection().feature_selection_TBFS(data, prediction_var, outcome_var, input_file)


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # get arguments from command line
    parser.add_argument("-f", "--filepath",
                        required=True,
                        default=None,
                        help="Path to target CSV file")

    # parse the argument from command line
    args = parser.parse_args()
    # Create instance of class
    objFS = Execute_feature_selection()
    # pass the command line argument to the method
    objFS.main_program(args.filepath)

    exit()