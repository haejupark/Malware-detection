import pandas as pd
import re
import os
from pandas import Series, DataFrame
from api_dict import *
from get_api_info import *
from get_permission_info import *

PATH_B = "test_apk/benign"
PATH_V = "test_apk/virus"
PATH_UV = "test_dex/unpack_virus"
PATH_UB = "test_dex/unpack_benign"

def main():
    fam = pd.read_csv('Malware_random_list_1st_500.csv')
    perdict_mal = get_permission_info(PATH_V)
    perdict_ben = get_permission_info(PATH_B)
    sus_dict_v = get_api_info(PATH_UV)
    sus_dict_b = get_api_info(PATH_UB)

    list_result = []

    # family add
    malware_dict = {}
    for i in fam['filename']:
        family_temp = fam[fam['filename']==i]['family']
        family_name = list(family_temp)[0]
        malware_dict[i] = {family_name}

    for key in perdict_ben:
        label = [0 for i in range(len(family)+1)]
        label[0] = 1
        list_result.append([key, perdict_ben[key], sus_dict_b[key],label])
        
    for key in perdict_mal:
        label = [0 for i in range(len(family)+1)]
        name = malware_dict[key].pop()
        if name in family:
            label[family.index(name)+1] = 1
        list_result.append([key, perdict_mal[key], sus_dict_v[key],label])
        
    frame = pd.DataFrame(list_result, columns= ['filename','permission','sus_b_c','label'],
                        index = range(len(list_result)))
#    display(frame)
    frame.to_csv("train.csv", sep=',')
    
main()
