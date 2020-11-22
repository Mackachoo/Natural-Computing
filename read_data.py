import matplotlib.pyplot as plt

data = [i.strip().split() for i in open("two_spirals.dat").readlines()]
for i in range(20):
    print(data[i][:2])

x = []
y = []

for i in data:
    x.append(float(i[0]))
    y.append(float(i[1]))

plt.scatter(x, y)
plt.show()