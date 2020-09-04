import pandas as pd
from collections import Counter

class Execute_GO_FreqCal():
    def frequncyCalculator(self, input_csv_file1):
        data_GO_IDs_pd = pd.read_csv(input_csv_file1)

        all_GO_IDs = []
        last_index_column = len(data_GO_IDs_pd.columns)

        # iterate for each rows
        for i, j in data_GO_IDs_pd.iterrows():
            # iterate for 2nd to last column of each rows
            for go_id in j[1:last_index_column].values:
                if str(go_id) != "nan":
                    all_GO_IDs.append(go_id)

        go_All_IDs_dataFrame = pd.DataFrame(data={"GO_IDs": all_GO_IDs})
        go_All_IDs_dataFrame.to_csv("All_GO_IDs_" + input_csv_file1, index=False)

        # this counter function will count occurrences of elements
        occurence_GO_IDs = Counter(all_GO_IDs)
        print(occurence_GO_IDs)
        occurence_GO_IDs_dataFrame = pd.DataFrame.from_dict(occurence_GO_IDs, orient='index').reset_index()
        occurence_GO_IDs_dataFrame = occurence_GO_IDs_dataFrame.rename(columns={'index': 'GO_IDs', 0: 'count'})
        occurence_GO_IDs_dataFrame.to_csv("GO_IDs_occurrences_" + input_csv_file1, index=False)

    def main_program(self, input_file1):
        Execute_GO_FreqCal().frequncyCalculator(input_file1)

if __name__=="__main__":

    user_input_file1 = input("Please enter the protein with Gene Ontology IDs data set name in csv format:: ")

    objEGOFC = Execute_GO_FreqCal()
    objEGOFC.main_program(user_input_file1)

    exit()