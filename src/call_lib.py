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
lac_lib = so('../lib/liblac.so')
print("cut_sentence")
s = '''
既往史：患者于4年前出现活动性心悸、胸痛，多在重体力活动时发作，胸痛位于剑突下和心前区，手掌大小，呈闷压样疼痛不适，每次持续10分钟左右，休息数分钟可缓解，发作时伴明显心悸、呼吸困难，无咳嗽、咳痰，无恶心、呕吐，无出汗，头晕、头痛。曾于2011年来我院就诊，诊断为“冠心病 不稳定心绞痛 房颤 心功能3级”，后正规服用药物，症状仍间断发作。3月来上述症状明显加重，表现为明显不能耐受体力活动，稍活动即有明显的胸痛发作，长舒气后症状有所缓解，伴四肢乏力，以双下肢为甚，伴夜间阵发性呼吸困难及端坐呼吸，上述症状间断出现，进行性加重，后出现双下肢水肿，晨轻暮重，今为进一步明确诊治，特来我院，门诊以“冠心病 心律失常 心功能不全”收入我科。
'''
lac_lib.freeme.argtypes = ctypes.c_char_p,
lac_lib.freeme.restype = None
lac_lib.cut_sentence.restype = ctypes.c_void_p
result = lac_lib.cut_sentence('../conf', 512, s)
print(type(result))
print(hex(result))
cut_result = ctypes.cast(result, ctypes.c_char_p).value
print("return result: " + cut_result)
lac_lib.freeme(result)
print(lac_lib.sum(1, 2))

