#   Version     Date            Comment
#       1.0     24/06/2021  Initial Release for single neuron

# NeuroNet(self, in_layer, n_layer, nodes)

import math
import random

class perceptron():
    weights = []
    type = None
    threshold = 0.5
    in_layer = []
    out_layer = 0
    output = 0
    eta = 0.1


    def __init__(self, weights, type = 'sigmoid', threshold=0, eta = .1):
        self.weights = weights
        self.type = type
        self.threshold = threshold
        self.eta = eta

    def compute(self,*args):
        if len(args) == 1:
            self.in_layer = args[0]
        elif len(args) == 2:
            self.in_layer = args[0]
            self.weights = args[1]
        elif len(args) == 3:
            self.in_layer = args[0]
            self.weights = args[1]
            self.threshold = args[2]
        else:
            print('More than 3 arguments passed')
            return -1

        if len(self.in_layer) + 1 == len(self.weights):
            l = len(self.in_layer)
            total_weight = [self.in_layer[i] * self.weights[i] for i in range(0,l)]
            total_weight = sum(total_weight) + self.weights[l]
            if self.type == 'sigmoid':
                out_layer = self.__sigmoid_func(total_weight)
                self.out_layer = out_layer
                return out_layer
            else:
                print("Function must be 'sigmoid' for now")
                return -1
        else:
            print('Number of input layer + 1 not equal to number of weights')
            return -1

    def __sigmoid_func(self,total_weight):
        out_layer = 1/(1+math.exp(-total_weight))
        return out_layer

    def bgp(self, output, err=0, isHidden=False):
        self.output = output
        l = len(self.weights) - 1
        del_w = []
        if isHidden == False:
            err = self.__err_func()
        for i in range(0,l):
            del_w.append(err * self.in_layer[i] * (1 - self.out_layer) * self.out_layer)
        del_w.append(err * (1 - self.out_layer) * self.out_layer)
        updated_w = []
        for i in range(0,l+1):
            updated_w.append(self.weights[i] - del_w[i] * self.eta)
        self.weights = updated_w
        #print('Updated weight: ', updated_w)
        return err * (1 - self.out_layer) * self.out_layer

    def __err_func(self):
        err_val = -(self.output - self.out_layer)
        return err_val

class NeuroNet():
    n_layer = 0
    in_layer = 0
    eta = 0
    nodes = []
    input_data = [] # store input data
    b = None        # store object of each perceptron
    o_layer = []    # data of output of each hidden layer
    output_data = []# store output data

    def __init__(self, in_layer, n_layer, nodes, eta = 0.1):
        if n_layer == len(nodes):
            self.n_layer = n_layer
            self.nodes = nodes
            self.in_layer = in_layer
            self.eta = eta
            self.__nodes_creation()
            #self.__test()
        else:
            print('Number of hidden layer should be equal to number of nodes given')

    def __nodes_creation(self):
        b = []
        for i in range(0, self.n_layer):
            a = []
            for j in range(0, self.nodes[i]):
                c = self.in_layer + 1 if i == 0 else self.nodes[i-1] + 1
                w = [random.random() * random.choice([1,-1]) for k in range(0,c)]
                a.append(perceptron(w, type = 'sigmoid', eta = self.eta))
            b.append(a)
        self.b = b
        #print(b)

    def __test(self):
        b = []
        a = []
        w = [-.1,.6,-.1]
        a.append(perceptron(w, type='sigmoid', eta=0.1))
        w = [.2,-.3,.3]
        a.append(perceptron(w, type='sigmoid', eta=0.1))
        b.append(a)
        a = []
        w = [.4,.5,.4]
        a.append(perceptron(w, type='sigmoid', eta=0.1))
        w = [.6,-.2,.2]
        a.append(perceptron(w, type='sigmoid', eta=0.1))
        b.append(a)

        self.b = b
        #print(b)

    def compute(self, input_data):
        self.input_data = input_data
        inp = input_data
        self.o_layer = []
        for i in range(0, self.n_layer):
            oup = []
            for j in self.b[i]:
                oup.append(j.compute(inp))
            inp = oup
            self.o_layer.append(oup)
        return self.o_layer[-1]

    def bgp(self, output_data):
        self.output_data = output_data
        l = len(self.b)
        err = []
        for i in range(l-1,-1,-1):
            a = []
            for j in range(0, len(self.b[i])):
                if err == []:
                    d = self.b[i][j].bgp(output_data[j])
                    a.append(d)
                else:
                    w = []
                    for k in self.b[i+1]:
                        w.append(k.weights[j])
                    err_w = [err[m] * w[m] for m in range(0,len(w))]
                    d = self.b[i][j].bgp(output_data, sum(err_w), True)
                    a.append(d)
            err = a

    def error(self, output_data):
        self.output_data = output_data
        err_val = [0.5 * (self.output_data[i] - self.o_layer[-1][i]) ** 2 for i in range(0, len(self.o_layer[-1]))]
        err_val = sum(err_val)
        print(err_val)
