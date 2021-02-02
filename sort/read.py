# Python code to
# demonstrate readlines()
 
# Using readlines()
file1 = open('tet.txt', 'r')
Lines = file1.readlines()
 
count = 0
all = []
# Strips the newline character
for line in Lines:
    generate = []
    count += 1
    result = line.split(' ')
    if len(result) != 7:
        break
    generate.append(result[2] + "_" + result[1] + "_" + result[3] )
    generate.append(result[5])
    print(generate)
    all.append(generate)


all = sorted(all, key=lambda x: x[0])
print(all)

for l in all:
    print(l)
