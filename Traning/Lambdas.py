from functools import reduce

#lambda
x = (lambda y, z: y + z)(5, 6)

#list initialazing
listOne = []
listTwo = []

for i in range(1, 100):
    listOne.append(i)
    listTwo.append(100 - i)


#filter
listOne = list(filter(lambda n: n < 10, listOne))
listTwo = list(filter(lambda n: n < 10, listTwo))


#map
listOne = list(map(lambda n: n * 10, listOne))
listTwo = list(map(lambda n: n * 100, listTwo))


#reduce
sumAll = reduce(lambda x, y: x + y, listOne)
print("sumAllFirst \t= ", sumAll)

sumAll = reduce(lambda x, y: x + y, listTwo)
print("sumAllSecond \t= ", sumAll)


maxEl = reduce(lambda x, y: x if x > y else y, listOne)
print("maxElFirts \t\t= ", maxEl)

maxEl = reduce(lambda x, y: x if x > y else y, listTwo)
print("maxElSecond \t= ", maxEl,"\n")


#zip
listOne.sort()
listTwo.sort()

diction = dict(zip(listOne, listTwo))

print("key\t\tvalue")
for key in diction.keys():
    print(key, "\t\t", diction.get(key))
