import pandas as pd
import re
import os
from pandas import Series, DataFrame
from api_dict import *
from get_api_info import *
from get_permission_info import *

PATH_S = "submit_apk"
PATH_US = "submit_dex"

def main():
    perdict_sub = get_permission_info(PATH_S)
    sus_dict_sub = get_api_info(PATH_US)
    
    list_result = []

    for key in perdict_sub:
        list_result.append([key, perdict_sub[key], sus_dict_sub[key]])
        
    frame = pd.DataFrame(list_result, columns= ['filename','permission','sus_b_c'],
                        index = range(len(list_result)))
    frame.to_csv("test.csv", sep=',')
    
main()
