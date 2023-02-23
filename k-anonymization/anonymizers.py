##############################################################################
# This skeleton was created by Mandana Bagheri (mmarzijarani20@ku.edu.tr)    #

# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
import glob
import os
import sys
from copy import deepcopy
from typing import Optional
import numpy as np
import copy
from tqdm import tqdm

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def extract_DGH_relationship(lines):
    lookup_table_DGH = {}

    def _recursive_DGH_tree(lines, parent, first_index, last_index):
        if first_index >= last_index:
            return
        
        current_parent_index=0

        for i in range(first_index, last_index):
            if lines[i].startswith('\t'):
                lines[i] = lines[i].replace('\t', '', 1)
            
            if not lines[i].startswith('\t'):
                if lines[i].rstrip(): #Some empty spaces exist in the DGH files. This line ensures that there exists some string.
                    lookup_table_DGH[lines[i].rstrip()] = parent.rstrip()
                if current_parent_index == 0:
                    current_parent_index=i
                else:
                    _recursive_DGH_tree(lines, lines[current_parent_index], current_parent_index+1, i)
                    current_parent_index=i
            
        
        _recursive_DGH_tree(lines, lines[current_parent_index], current_parent_index+1, last_index)
                    
    lookup_table_DGH["most_general"] = lines[0].rstrip()
    _recursive_DGH_tree(lines, lines[0], 1, len(lines))
    return lookup_table_DGH


def create_depth_lookup_table(DGHs):
    depth_table = {}
    for relation_dictionary in DGHs.values():
        most_general = relation_dictionary['most_general']        
        del relation_dictionary['most_general']

        for child in relation_dictionary:
            depth = 1
            parent = relation_dictionary[child]

            while parent != most_general:
                parent = relation_dictionary[parent]
                depth += 1

            depth_table[child] = depth
        
        depth_table[most_general] = 0
    return depth_table

def create_descendant_leaves_lookup_table(DGHs):
    def calculate_descendant_leaves(list_of_leaves, number_of_descendant_leaves, relation_dictionary, target):
        if number_of_descendant_leaves[target] != 0:
            return number_of_descendant_leaves[target]
        else:
            result = 0
            for child, parent in relation_dictionary.items():
                if parent == target:
                    result += calculate_descendant_leaves(list_of_leaves, number_of_descendant_leaves, relation_dictionary, child)
            return result

    lookup_table = {}
    total_number_of_childs = {}
    for attribute, relation_dictionary in DGHs.items():
        most_general = relation_dictionary["most_general"]
        del relation_dictionary['most_general']
        parent = set(relation_dictionary.values())
        children = set(relation_dictionary.keys())
        list_of_leaves = list(children - parent)
        total_number_of_childs[attribute] = len(list_of_leaves)
        number_of_descendant_leaves = {key:0 for key in relation_dictionary}
        number_of_descendant_leaves[most_general] = 0
        for leave in list_of_leaves:
            number_of_descendant_leaves[leave] = 1

        for key in number_of_descendant_leaves:
            number_of_descendant_leaves[key] = calculate_descendant_leaves(list_of_leaves, number_of_descendant_leaves, relation_dictionary, key)
        
        lookup_table.update(number_of_descendant_leaves)
    
    return lookup_table, total_number_of_childs


def find_number_of_QIs(DGHs, row):
    result = 0
    for key in row:
        if key in DGHs.keys():
            result += 1
    return result
    
def find_QIs(DGHs, row):
    QIs = []
    for attribute in row:
        if attribute in list(DGHs.keys()):
            QIs.append(attribute)
    return QIs

def generalize_single_attribute(cluster, attribute, DGHs):
    for i in range(len(cluster)):
        try:
            cluster[i][attribute] = DGHs[attribute][str(cluster[i][attribute])]
        except KeyError:
            continue #If it is already most generalized version (e.g "Any"), it doesn't have a parent.

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]['age']) # debug: testing.
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    with open(DGH_file) as f:
        return extract_DGH_relationship(f.read().split("\n"))



def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}

    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs

def calculate_MD_cost(raw_dataset: list, 
                      anonymized_dataset: list, 
                      QIs: dict, 
                      depth_lookup_table: dict) -> float:
    MD_cost = 0
    for row in range(len(raw_dataset)):
        for key in raw_dataset[row].keys():
            if key in QIs:
                MD_cost += depth_lookup_table[raw_dataset[row][key]] - depth_lookup_table[anonymized_dataset[row][key]]
    #TODO: complete this function.
    return MD_cost

def calculate_LM_cost(anonymized_dataset, QIs, descendant_leaves_lookup_table, total_number_of_childs):
    number_of_QIs = len(QIs)
    LM_cost = 0
    for row in range(len(anonymized_dataset)):
        for attribute, value in anonymized_dataset[row].items():
            if attribute in QIs:
                LM_cost += ((descendant_leaves_lookup_table[value] - 1) / (total_number_of_childs[attribute] - 1)) * (1/number_of_QIs)

    return LM_cost

def create_EC(DGHs, QIs, cluster):
    for attribute in QIs:
            done_flag = False
            while True:
                for i in range(len(cluster)-1):
                    if cluster[i][attribute] == cluster[i+1][attribute]:
                        done_flag = True
                        continue
                    else:
                        generalize_single_attribute(cluster, attribute, DGHs)
                        done_flag = False
                        break #break the for loop


                if i == len(cluster)-2 and done_flag:
                    break #break the while loop
    return cluster

def calculate_distance(row1, row2, DGHs, QIs, depth_lookup_table):
    hypothetical_cluster = [row1.copy(), row2.copy()]
    create_EC(DGHs, QIs, hypothetical_cluster)
    distance = calculate_MD_cost([row1, row2] ,hypothetical_cluster, QIs, depth_lookup_table)
    return distance

def rows_equal(row1, row2, QIs):
    for attribute in QIs:
        if row1[attribute] != row2[attribute]:
            return False
    return True

def specialize_dataset(raw_dataset, dataset, DGHs, indices, attribute):
    for i in indices:
        value = raw_dataset[i][attribute]
        while DGHs[attribute][value] != dataset[i][attribute] and dataset[i][attribute] != value:
            value = DGHs[attribute][value]
        dataset[i][attribute] = value
        
def find_ECs(dataset, indices, QIs):
    ECs = []
    for i in indices:
        new_row_flag = True
        for class_indices in ECs:
            if rows_equal(dataset[class_indices[0]], dataset[i], QIs):
                class_indices.append(i)
                new_row_flag = False
                break
        if new_row_flag:
            ECs.append([i])
    return ECs

def recursive_topdown(dataset, raw_dataset, k, descendant_leaves_lookup_table, total_number_of_childs, indices, QIs, DGHs):
    possible_splits = []
    for attribute in QIs:
        hypotetical_dataset = copy.deepcopy(dataset)
        specialize_dataset(raw_dataset, hypotetical_dataset, DGHs, indices, attribute)
        ECs = find_ECs(hypotetical_dataset, indices, QIs)
        if all(len(row) > k for row in ECs) and len(ECs) > 1:
            LM_cost = calculate_LM_cost(hypotetical_dataset[indices], QIs, descendant_leaves_lookup_table, total_number_of_childs)
            possible_splits.append((attribute, LM_cost))
    
    del hypotetical_dataset

    if possible_splits:
        selected_attribute = min(possible_splits, key = lambda pair: pair[1])[0]
        specialize_dataset(raw_dataset, dataset, DGHs, indices, selected_attribute)
        ECs = find_ECs(dataset, indices, QIs)
        for class_indices in ECs:
            recursive_topdown(dataset, raw_dataset, k, descendant_leaves_lookup_table, total_number_of_childs, class_indices, QIs, DGHs)

def generalize_fully(QIs, DGHs, dataset):
    generalized = copy.deepcopy(dataset)
    for i in range(len(dataset)):
        for attribute in DGHs:
            while generalized[i][attribute] != DGHs[attribute]["most_general"]:
                generalized[i][attribute] = DGHs[attribute][str(generalized[i][attribute])]
    return np.asarray(generalized)
##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    depth_lookup_table = create_depth_lookup_table(DGHs)
    QIs = find_QIs(DGHs, raw_dataset[0])

    
    #TODO: complete this function.

    return calculate_MD_cost(raw_dataset, anonymized_dataset, QIs, depth_lookup_table)


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    descendant_leaves_lookup_table, total_number_of_childs = create_descendant_leaves_lookup_table(DGHs)
    QIs = find_QIs(DGHs, raw_dataset[0])

 
    #TODO: complete this function.
    return calculate_LM_cost(anonymized_dataset, QIs, descendant_leaves_lookup_table, total_number_of_childs)


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    QIs = find_QIs(DGHs, raw_dataset[0])

    #Divide into clusters of size k.
    arr = np.asarray(raw_dataset)
    length = len(raw_dataset)
    normalized_length = length - (length % k) - k
    clusters = np.split(arr[0:normalized_length], normalized_length/k)
    clusters.append(arr[normalized_length:])

    for cluster in clusters:
        create_EC(DGHs, QIs, cluster)

    anonymized_dataset = []
    for cluster in clusters:
        for entry in cluster:
            anonymized_dataset.append(entry)
    #TODO: complete this function.

    write_dataset(anonymized_dataset, output_file)


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    length = len(raw_dataset)
    dataset = np.asarray(raw_dataset)[:length]
    scores = np.zeros((length))
    QIs = find_QIs(DGHs, raw_dataset[0])
    depth_lookup_table = create_depth_lookup_table(DGHs)

    #TODO: complete this function.

    for iteration in tqdm(range(length // k)):

        topmost_index = np.where(scores != np.inf)[0][0]
        scores[topmost_index] = np.inf

        for i in np.where(scores != np.inf)[0]:
            scores[i] = calculate_distance(raw_dataset[topmost_index], raw_dataset[i], DGHs, QIs, depth_lookup_table)


        closest_indices = np.argpartition(scores, k-1)[: k-1]
        closest_indices = np.append(closest_indices, topmost_index)
        equivalence_class = copy.deepcopy(dataset[closest_indices])

        create_EC(DGHs, QIs, equivalence_class) 
        scores[closest_indices] = np.inf
        dataset[closest_indices] = equivalence_class
        
    #Resolve unmarked results
    last_indices = np.append(closest_indices, np.where(scores != np.inf))
    equivalence_class = copy.deepcopy(dataset[last_indices])
    create_EC(DGHs, QIs, equivalence_class) 
    scores[last_indices] = np.inf
    dataset[last_indices] = equivalence_class


    write_dataset(dataset.tolist(), output_file)


def topdown_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Top-down anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    QIs = find_QIs(DGHs, raw_dataset[0])
    length = len(raw_dataset)

    dataset = generalize_fully(QIs, DGHs, raw_dataset)[0:length]
    descendant_leaves_lookup_table, total_number_of_childs = create_descendant_leaves_lookup_table(DGHs)
    recursive_topdown(dataset, raw_dataset, k, descendant_leaves_lookup_table, total_number_of_childs, range(length), QIs, DGHs)    
    #TODO: complete this function.

    write_dataset(dataset.tolist(), output_file)

# # Command line argument handling and calling of respective anonymizer:
# if len(sys.argv) < 6:
#     print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
#     print(f"\tWhere algorithm is one of [clustering, random, topdown]")
#     sys.exit(1)

# algorithm = sys.argv[1]
# if algorithm not in ['clustering', 'random', 'topdown']:
#     print("Invalid algorithm.")
#     sys.exit(2)

# dgh_path = sys.argv[2]
# raw_file = sys.argv[3]
# anonymized_file = sys.argv[4]
# k = int(sys.argv[5])

# function = eval(f"{algorithm}_anonymizer");
# function(raw_file, dgh_path, k, anonymized_file)

# cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
# cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
# print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")


# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300
