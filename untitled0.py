# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 18:28:08 2018

@author: 11854
"""

import numpy as np
import matplotlib.pyplot as plt

y = np.arange(35)
b = y>20
#b是bool类型
# In[]:
y_range=np.arange(35).reshape(5,7)
#生成一个非0数组的方法
y_range[np.array[0,2,4],np.array[0,1,2]]
#选取0,2,4行，0,1,2列的数

#布尔值或者掩码索引数组
