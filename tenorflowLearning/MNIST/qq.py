# -*- coding: utf-8 -*-
'''
@Author  : Arron
@email   :hou.zg@foxmail.com
@software: python
@File    : qq.py
@Time    : 2018/2/26 1:37
'''
class A:
    @classmethod
    def a(cls):
        return 2
    @classmethod
    def b(cls):
        return cls.a()**3
    c=b()
a=A()

