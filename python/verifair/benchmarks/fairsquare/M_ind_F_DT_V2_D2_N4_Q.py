from .helper import *

def sample(flag):
    age = gaussian(38.5816, 186.0614)
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    capital_gain = gaussian(1077.6488, 54542539.1784)
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    if capital_gain >= 7073.5:
        if age < 20:
            t = 1
        else:
            t = 0
    else:
        t = 1
    return int(t < 0.5)
    fairnessTarget(t < 0.5)

