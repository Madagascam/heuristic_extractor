a = [1, 2, 3]

def check(b):
    b[0] = -2
    print(b)
    return(b)

c = check(a)
print(c)
print(a)
