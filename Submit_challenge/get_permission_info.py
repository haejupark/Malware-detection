import os
from permissionlist import *
from apk_parse.apk import APK
def get_permission_info(PATH):
    filenames = os.listdir(PATH)
    get_permission = {}
    my_list = [0 for i in range(len(permission_v2))]
    for filename in filenames:
        # add permission list 11/12
        my_list = [0 for i in range(len(permission_v2))]
        
        full_path = PATH + "/" + filename
        apkf = APK(full_path)
        permission_list = []
        count = 0
        for i in apkf.get_permissions():
            permission_list.append(i.split(".")[-1])

        for i in permission_list:
            if i in permission_v2:
                my_list[permission_v2.index(i)]=1
                count += 1
        my_list.append(count)
        get_permission[filename[:-4]]= my_list
    return get_permission
