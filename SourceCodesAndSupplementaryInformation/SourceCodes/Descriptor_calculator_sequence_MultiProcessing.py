import re
from pydpi.pypro import PyPro
import pandas as pd
from multiprocessing import Pool

# this function will not work under class.....
def seq_multiprocess(seq_items):
    try:
        return Execute_descriptor_generator().extract_value_ds(
            Execute_descriptor_generator().descriptor_generator(seq_items))
    except:
        print("Error Occurece....")
        pass

class Execute_descriptor_generator():
    def read_fasta(self, input_fasta):
        fasta_header = []
        fasta_sequences = []
        temp_sequence = None

        file_fasta = open(input_fasta)
        file_lines = file_fasta.readlines()

        for line in file_lines:
            line = line.rstrip()
            if line.startswith(">"):
                # this use to extract uniprot id from header
                uniprot_id_search = re.search(">sp\|(.*)\|", line)
                if uniprot_id_search:
                    fasta_header.append(uniprot_id_search.group(1))
                else:
                    fasta_header.append("Not Found Pattern")
                # this section use to append all sequence expect last
                if temp_sequence:
                    # these replace are use to few error AA to known ones
                    temp_seq_replace1 = temp_sequence.replace("U","A")
                    temp_seq_replace2 = temp_seq_replace1.replace("X","C")
                    temp_seq_replace3 = temp_seq_replace2.replace("B","D")
                    temp_seq_replace4 = temp_seq_replace3.replace("J","E")
                    temp_seq_replace5 = temp_seq_replace4.replace("Z","F")
                    temp_seq_replace6 = temp_seq_replace5.replace("O", "G")
                    fasta_sequences.append(temp_seq_replace6)

                    # fasta_sequences.append(temp_sequence)
                temp_sequence = ''
            else:
                temp_sequence += line
        # this section use to append last sequence
        if temp_sequence:
            # these replace are use to few error AA to known ones
            temp_seq_replace1 = temp_sequence.replace("U", "A")
            temp_seq_replace2 = temp_seq_replace1.replace("X", "C")
            temp_seq_replace3 = temp_seq_replace2.replace("B", "D")
            temp_seq_replace4 = temp_seq_replace3.replace("J", "E")
            temp_seq_replace5 = temp_seq_replace4.replace("Z", "F")
            temp_seq_replace6 = temp_seq_replace5.replace("O", "G")
            fasta_sequences.append(temp_seq_replace6)

            # fasta_sequences.append(temp_sequence)

        # print("UniprotId:",fasta_header,"Sequence:", fasta_sequences)
        return fasta_header, fasta_sequences

    def descriptor_generator(self, protein_sequence):
        obj_PyPro = PyPro()
        obj_PyPro.ReadProteinSequence(protein_sequence)

        ds1 = obj_PyPro.GetAAComp()
        ds2 = obj_PyPro.GetDPComp()
        ds3 = obj_PyPro.GetPAAC(lamda=30)
        ds4 = obj_PyPro.GetCTD()
        ds5 = obj_PyPro.GetQSO()
        ds6 = obj_PyPro.GetTriad()

        ds_all = []
        # This is use to append since sequentially add .. otherwise update() function update not sequentially
        for ds in (ds1, ds2, ds3, ds4, ds5, ds6):
            ds_all.append(ds)

        return ds_all

    def extract_header_ds(self, list_dictionary):
        header_ds = []

        for index, dictionary in enumerate(list_dictionary):
            for dict_key in dictionary.keys():
                header_ds.append(dict_key)

        # print(header_ds)
        return header_ds

    def extract_value_ds(self, list_dictionary):
        value_ds = []

        for index, dictionary in enumerate(list_dictionary):
            for dict_value in dictionary.values():
                value_ds.append(dict_value)

        # print(value_ds)
        return value_ds

    def main_program(self, input_file):
        self.input_file = input_file
        uniprot_ids, sequences = Execute_descriptor_generator().read_fasta(self.input_file)

        # this is used to get header name for different sequence compositions features
        ds_header =  Execute_descriptor_generator().extract_header_ds(
            Execute_descriptor_generator().descriptor_generator(sequences[0]))



        max_seq = len(sequences)

        print("Total number of sequences:", max_seq)

        # Multiprocessing pool start here.....
        pool = Pool(processes=3)
        ds_values = pool.map(seq_multiprocess, sequences)
        print("Total process proteins:::", len(ds_values))

        result_ds = pd.DataFrame(data=ds_values, columns=ds_header)
        # this will add a UniprotId column to dataframe
        result_ds.insert(loc=0, column="UniprotId", value=uniprot_ids)

        result_ds.to_csv("Descriptors_MP_" + str(input_file).replace(".fasta", ".csv"), index=False)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # get arguments from command line
    parser.add_argument("-f", "--filepath",
                        required=True,
                        default=None,
                        help="Path to target fasta file")
    args = parser.parse_args()
    # Create instance of class
    objEDG = Execute_descriptor_generator()
    # pass the command line argument to the method
    objEDG.main_program(args.filepath)

    exit()
