import math
def Rmse(res):
    error=0
    count=0
    for i in res:
        error+=abs(i[2]-i[3])
        count+=1
    if count == 0:
        return error
    return math.sqrt(error)

def Mae(res):
    error=0
    count=0
    for i in res:
        error+=abs(i[2]-i[3])
        count+=1
    if count==0:
        return error
    return float(error)/count