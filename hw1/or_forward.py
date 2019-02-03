#!/usr/bin/python3

"""Script to build a feedforward OR gate"""


class Neuron(object):
    """A Neuron object. It contains inputs, weights and an activation function"""

    def __init__(self, inp_lis, w_lis):
        """Constructor"""
        self.y = 0
        self.out = 0
        self.inp_list = inp_lis
        self.weight_list = w_lis

    def binary_func(self):
        """Returns 1 if the weighted sum of inputs is >0, else -1"""
        return -1 if self.y < 0 else 1

    def activation_function(self):
        """Computes the output of the neuron"""
        self.out = self.binary_func()

    def calc_net(self):
        """Calculate the weighted sum of inputs"""
        net = 0
        out_lis = [a * b for a, b in zip(self.inp_list, self.weight_list)]
        self.y = sum(out_lis)

    def forward_prop(self):
        """Does one forward prop through the neuron"""
        self.calc_net()
        self.activation_function()

    def display_stuff(self):
        """Displays the neurons, the weights, the net and the output"""
        print("i1 | i2 | b  | Net| Output")
        print(*self.inp_list, self.y, self.out, sep="  | ")
        print("Weights used are: ")
        print(*self.weight_list)


def main():
    """The main method"""
    print("Building an OR gate using neural networks")
    print("There is no training involved, just the final weights that give the correct output")
    print("Using binary function as the activation function")
    ws = [2, 2, -1]
    is1 = [1, 1, 1]
    is2 = [1, 0, 1]
    is3 = [0, 1, 1]
    is4 = [0, 0, 1]
    n1 = Neuron(is1, ws)
    n1.forward_prop()
    n1.display_stuff()

    n2 = Neuron(is2, ws)
    n2.forward_prop()
    n2.display_stuff()

    n3 = Neuron(is3, ws)
    n3.forward_prop()
    n3.display_stuff()

    n4 = Neuron(is4, ws)
    n4.forward_prop()
    n4.display_stuff()


if __name__ == "__main__":
    main()
