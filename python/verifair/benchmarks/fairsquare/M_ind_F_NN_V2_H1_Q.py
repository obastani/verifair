from .helper import *

def sample(flag):
    age = gaussian(38.5816, 186.0614)
    education_num = gaussian(10.0806, 6.6188)
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    N_age = ((age - 17.0) / 73.0  - 0.5) * 10 + 0.5
    N_education_num = ((education_num - 3.0) / 13.0  - 0.5) * 10 + 0.5
    h1 =  0.1718 * N_age +  1.1416 * N_education_num +  0.4754
    if h1 < 0:
        h1 = 0
    o1 =  0.4778 * h1 +  1.2091
    if o1 < 0:
        o1 = 0
    o2 =  1.9717 * h1 + -0.3104
    if o2 < 0:
        o2 = 0
    return int(o1 < o2)
    fairnessTarget(o1 < o2)

