import os
import math
import re

os.system("cls")

print("\n\nNEW INPUT\n")

def arrayToInt(array): return [int(i) for i in array]

def inc2dArray(array, x = 1): return [[j + x for j in i] for i in array]

inputNum = []

with open("input.txt", "r") as fp:
    for i in fp.read().split("\n"): 
        inputNum.append(i)

print(inputNum)