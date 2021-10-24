from package.NeuroNet import perceptron, NeuroNet
import random

if __name__ == '__main__':
    # a = NeuroNet(2,2,[2,2])
    # b = a.compute([0,1])
    # print(b)
    # a.bgp([0, 1])
    # print('complete')

    print('Welcome')
    import csv

    with open('data/iris.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)[1:]

    random.shuffle(data)
    a = NeuroNet(4, 2, [4, 3], 0.1)
    for m in range(0,200):
        for i in range(0,140):
            row_data = [float(j) for j in data[i]]
            p = a.compute(row_data[:-3])
            print(row_data[:-3])
            print(p)
            print(row_data[-3:])
            a.bgp(row_data[-3:])
            a.error(row_data[-3:])
            print()
    print('\n\n\n\nNow it will work for previously unseen data')
    for i in range(140,150):
        row_data = [float(j) for j in data[i]]
        p = a.compute(row_data[:-3])
        print(row_data[:-3])
        if p[0] == max(p):
            p = 'Iris-setosa'
        elif p[1] == max(p):
            p = 'Iris-versicolor'
        elif p[2] == max(p):
            p = 'Iris-virginica'
        print('Predicted: ',p)
        if row_data[-3] == max(row_data[-3:]):
            x = 'Iris-setosa'
        elif row_data[-2] == max(row_data[-3:]):
            x = 'Iris-versicolor'
        elif row_data[-1] == max(row_data[-3:]):
            x = 'Iris-virginica'
        print('Actual: ',x)
        a.error(row_data[-3:])
        print()
    print(data)