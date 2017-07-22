def sum(n):
    if n == 0 :
        return 0
    return n + sum(n-1)
    """
    :type n: int
    :rtype: int
    """
    # Fill out,  Use recursion


def fibonacci(n):
    if n == 1 or n == 2 :
        return 1
    return fibonacci(n-1) + fibonacci(n-2)
    """:
    :type n: int
    :rtype: int
    """
    # Fill out,  Use recursion


def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
    """
    :type n: int
    :rtype: int
    """


# Fill out,  Use recursion


def decimal_to_binary(n):
    if n < 2:
        return n
    else:
        return decimal_to_binary(n//2)*10+(n%2)



    """:
    :type n: int
    :rtype: int
    """
    # Fill out,  Use recursion


def TestRecursionFunction():
    print factorial(10)
    print sum(100)
    print fibonacci(10)
    print decimal_to_binary(15)


TestRecursionFunction()
