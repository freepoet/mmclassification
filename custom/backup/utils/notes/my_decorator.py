import time


def display_time(func):
    def wrapper(*args):
        t1 = time.time()
        result=func(*args)
        t2 = time.time()
        print("total time:{:.4f}  s".format(t2 - t1))
        return result
    return wrapper

def is_prime(num):
    if num<2:
        return False
    elif num==2:
        return True
    else:
        for i in range(2,num):
            if num%i==0:
                return  False
        return True
@display_time
def count_prime_nums(maxnum):
    count=0
    for i in range(2,maxnum):
        if is_prime(i):
            count = count +1
    return count
maxnum=10000
count=count_prime_nums(maxnum)
print(count)