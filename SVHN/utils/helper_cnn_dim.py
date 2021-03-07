#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:29:34 2020

@author: guillermogarcia
"""

def outputsize(in_size,kernel_size,stride,padding):
  output = int((in_size - kernel_size + 2*(padding))/stride) +1
  return output



