import pandas as pd

class Execute_Top_GO_Presence():
    def go_Presence(self, input_csv_file1, input_csv_file2):
        top_GO_IDs_pd = pd.read_csv(input_csv_file1)
        # print(list(top_GO_IDs_pd[top_GO_IDs_pd.columns[0]]))

        columnName = list(top_GO_IDs_pd[top_GO_IDs_pd.columns[0]])
        columnName.insert(0, "UniprotID")
        # print(columnName)

        protein_with_GO_IDs_pd = pd.read_csv(input_csv_file2)
        last_index_column = len(protein_with_GO_IDs_pd.columns)
        # print(protein_with_GO_IDs_pd)
        output_list = []

        # iterate for each rows
        for i, j in protein_with_GO_IDs_pd.iterrows():
            # output_list.insert(0, j[0])
            zeros_list = [0] * len(list(top_GO_IDs_pd[top_GO_IDs_pd.columns[0]]))
            zeros_list.insert(0, j[0])
            output_list.append(zeros_list)
            # iterate for 2nd to last column of each rows
            for go_id in j[1:last_index_column].values:
                if str(go_id) != "nan":
                    if go_id in list(top_GO_IDs_pd[top_GO_IDs_pd.columns[0]]):
                        find_go_id_index = list(top_GO_IDs_pd[top_GO_IDs_pd.columns[0]]).index(go_id)
                        output_list[i][find_go_id_index+1] = 1

        protein_with_TopGO_indicator = pd.DataFrame(columns=columnName, data=output_list)
        protein_with_TopGO_indicator.to_csv("TopGO_Indicator_" + input_csv_file2, index=False)

    def main_program(self, input_file1, input_file2):
        Execute_Top_GO_Presence().go_Presence(input_file1, input_file2)

if __name__=="__main__":

    user_input_file1 = input("Please enter the TOP Gene Ontology IDs data set name in csv format:: ")
    user_input_file2 = input("Please enter the protein with Gene Ontology IDs data set name in csv format:: ")

    objTGOP = Execute_Top_GO_Presence()
    objTGOP.main_program(user_input_file1, user_input_file2)

    exit()