#!/usr/bin/env python
# -*- coding:utf-8 -*-
a, b, c, ab, bc, ac, abc = 120, 140, 120, 170, 190, 160, 255
x = [('a', 120), ('b', 140), ('c', 120), ('ab', 170)]
for i in x:
    if 'a' in i[0]:
        print(i[1])
