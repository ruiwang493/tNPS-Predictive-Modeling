# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 09:21:28 2022

@author: RUI.WANG2
"""

import sys
import os
import logging

# package library
sys.path.insert(0, os.getenv("USERPROFILE") + "\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\pkg")

import glob_var
glob_var.init()

import functions as func


logging.basicConfig(
    filename=glob_var.log,
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    )
logger = logging.getLogger(__name__)