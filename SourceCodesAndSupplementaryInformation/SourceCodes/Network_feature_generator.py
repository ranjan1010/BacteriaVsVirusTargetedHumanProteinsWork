import csv
import pandas as pd
class Execute_network_feature_generator():
    def network_feature_generator(self, input_csv_file1, input_csv_file2):
        uniprotID_list = list(csv.reader(open(input_csv_file1)))
        # uniprotID_header = uniprotID_list[0]
        uniprotID_values = uniprotID_list[1 : len(uniprotID_list)]

        print("Total protein ids in UniprotID file is ::", (len(uniprotID_list) - 1))

        # pandas use to handle data in table format
        uniprotID_to_network_feature = pd.read_csv(input_csv_file2)

        last_index_column = len(uniprotID_to_network_feature.columns)
        output_list = []
        output_list.append(uniprotID_to_network_feature.columns[0:last_index_column])

        # this converted all values in list format in order to search value in list
        uniprotID_to_network_feature_values = list(uniprotID_to_network_feature[uniprotID_to_network_feature.columns[0 : last_index_column]].values)
        for uniprot_id in uniprotID_values:
                #  this will check if uniprotId in uniprotID_to_network_feature list
                if uniprot_id[0] in list(uniprotID_to_network_feature[uniprotID_to_network_feature.columns[0]].values):
                        # this will help to find first match value based on the first column
                        output_list.append(uniprotID_to_network_feature_values[list(uniprotID_to_network_feature[uniprotID_to_network_feature.columns[0]].values).index(uniprot_id[0])])
                else:
                    # if not find in network feature file then assign zeros to all features corresponding to the uniprotId
                    zeros_list = [0] * (last_index_column - 1)
                    # insert uniprotId in the first position of the list
                    zeros_list.insert(0, uniprot_id[0])
                    # finally append this list to the ouput list
                    output_list.append(zeros_list)

        print("Total protein ids in output file is ::", (len(output_list) - 1))

        output_file = open( "NetworkProperties_"+ str(input_csv_file1), "wb")
        csv_writer =csv.writer(output_file)
        csv_writer.writerows(output_list)
        output_file.close()


    def main_program(self, input_file1, input_file2):
        Execute_network_feature_generator().network_feature_generator(input_file1,input_file2)

if __name__=="__main__":

    user_input_file1 = raw_input("Please enter the only UniprotID file name in csv format:: ")
    user_input_file2 = raw_input("Please enter the UniprotID to Network Features file name in csv format:: ")

    objENFG = Execute_network_feature_generator()
    objENFG.main_program(user_input_file1, user_input_file2)

    exit()