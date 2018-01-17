import array
import timeit

list_1 = [i for i in range(0, 1000)]
array_1 = array.array('l', list_1)

def list_sum():
    sum = 0
    for i in list_1:
        sum += i
    return sum

def arr_sum():
    sum = 0
    for i in array_1:
        sum += i
    return sum

def list_increment():
    for i in range(len(list_1)):
        list_1[i] += 1

def arr_increment():
    for i in range(len(array_1)):
        array_1[i] += 1


list_sum_time = timeit.timeit(list_sum, number=10000)
arr_sum_time = timeit.timeit(arr_sum, number=10000)
list_increment_time = timeit.timeit(list_increment, number=10000)
arr_increment_time = timeit.timeit(arr_increment, number=10000)

print("list_sum_time={}".format(list_sum_time))
print("arr_sum_time={}".format(arr_sum_time))
print("list_increment_time={}".format(list_increment_time))
print("arr_increment_time={}".format(arr_increment_time))

