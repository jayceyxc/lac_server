#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-11-30 16:54
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : call_lib.py.py
# @Software: PyCharm
# @Description 测试调用C++库

import ctypes

so = ctypes.cdll.LoadLibrary
lac_lib = so('../lib/liblac_shared.so')
print("lac_create")
lac_lib.lac_create('../conf')
