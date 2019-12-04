#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil

def clear_dir(directory):
    """
    Removes all files in the given directory.
    """
    # important! if None passed to os.listdir, current directory is wiped (!)
    if not os.path.isdir(directory): raise Exception("%s is not a directory"%(directory))
    if type(directory) != str: raise Exception("string type required for directory: %s"%(directory))
    if directory in ["..",".", "","/","./","../","*"]: raise Exception("trying to delete current directory, probably bad idea?!")
    
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)
            
            
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def isLocked(path):
    if os.path.exists(path):
        return True
    try:
        os.mkdir(path)
        return False
    except:
        return True
    
def unlock(path):
    os.rmdir(path)
