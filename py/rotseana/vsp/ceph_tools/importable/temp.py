from lccal import lccal
from unconex import unconex

ans = unconex(lccal(27.89021,36.39717,0.1,False),'sn')
print('UNCONEX')
for i in ans:
    print(i[0],i[1],i[2])
