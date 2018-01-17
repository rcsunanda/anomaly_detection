
def func(arg1, *arg2):
    print("arg1={}, type_of_arg2={}".format(arg1, type(arg2)))
    for a in arg2:
        print("\t a={}".format(a))


func("a1", "v1", "v2", "v3")
# func("a1", *("v1", "v2", "v3"))

# ---------------------------------------
print("--" * 40)

def func2(arg1, **kwargs):
    print("arg1={}, type_of_kwargs={}".format(arg1, type(kwargs)))
    for k,v in kwargs.items():
        print("\t k={}, v={}".format(k, v))


func2("a1", arg2="v1", arg3="v2", arg4="v3")
# func2(arg1="a1", arg2="v1", arg3="v2", arg4="v3")

# ---------------------------------------
print("--" * 40)

def func1(val):
    print("val={}, const={}".format(val, CONST))

def func4():

    def func3(val):
        some_local = 1
        print("val={}, const={}, some_local={}".format(val, CONST, some_local))

    CONST = 600
    func3(3)

# CONST = 500
func4()


# ---------------------------------------
print("--" * 40)

def func1(val):
    print(val)
    # print(other_val)

def func2():
    other_val = 2
    func1(100)

func2()



# ---------------------------------------
print("--" * 40)


def func1():
    print(g_var)    # Looks up LEGB in order and finds in the global namespace; prints 10
                    # if g_var were mutable, can mutate it as well

g_var = 10
func1()

####

def func2():
    g_var = 20  # Adds a new name to the local namespace (referring to a new object int(20))
                # The global variable g_var is no longer accessible in this function
    print(g_var)    # Prints 20

g_var = 10
func2()
print(g_var)  # Prints 10 (the global object was not affected in the function)

####
print("@" * 10)

def func3():
    global g_var
    g_var = 20  # Replaces the global name binding with a new object int(20)
    print(g_var)    # Prints 20

g_var = 10
func3()
print(g_var)  # Prints 20 (the global name was set to a new int(20) in the function)
