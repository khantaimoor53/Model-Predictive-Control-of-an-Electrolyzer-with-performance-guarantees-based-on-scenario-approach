# %%
import numpy as np
from matplotlib import pyplot as plt


# The PDF of the M-shaped distribution
# 
# x is from 0.8 to 1.2

def PDF_m_shaped(a,mid,b,y_max,x):
    
    if a <= x <= mid:
        return ((y_max-2)/(a-mid))*(x-a) + y_max
    
    elif mid < x <= b:
        return ((2-y_max)/(mid-b))*(x-b) + y_max

    
# %%
def CDF_m_shaped(a,mid,b,y_max,x):

    m_1 = (y_max-2)/(a-mid)
    m_2 = (2-y_max)/(mid-b)

    if a <= x <= mid:
        return ((m_1*(((x**2)/2)-(a*x))) + y_max * x) - ((m_1*(((0.8**2)/2)-(a*0.8))) + y_max * 0.8)
    
    elif mid < x <= b:
        return 0.5 + (((m_2*(((x**2)/2)-(b*x))) + y_max * x) - ((m_2*(((1**2)/2)-(b*1))) + y_max * 1))
# %%

def inverse_CDF_m_shaped(y):

    # y = np.random.uniform(0,1)
    # y = round(y,3)

    if y <= 0.5:
        return (-1*np.sqrt(-0.4*y+0.36)+1.4)
    
    else:
        return (1*np.sqrt(0.4*y-0.04)+0.6)
    
   
# %%
