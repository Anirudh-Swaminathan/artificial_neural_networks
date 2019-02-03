#!/usr/bin/python3

"""Script to train a single-layer perceptron network to learn the XOR function"""


class Neuron(object):
    """A Neuron object. It contains inputs, weights and an activation function"""

    def __init__(self, inp_lis, w_lis, thresh):
        """Constructor"""
        self.y = 0
        self.out = 0
        self.inp_list = inp_lis
        self.weight_list = w_lis
        self.threshold = thresh

    def binary_func(self):
        """Returns 1 if the weighted sum of inputs is >0, else -1"""
        return -1 if self.y < 0 else 1

    def perceptron_func(self):
        """Returns 1 if weighted sum of inputs > threshold, 0 if between negative and positive of threshold, and 0
        if less than negative threshold
        """
        if self.y > self.threshold:
            return 1
        elif -1 * self.threshold <= self.y <= self.threshold:
            return 0
        else:
            return -1

    def activation_function(self):
        """Computes the output of the neuron"""
        self.out = self.perceptron_func()

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
        # print("i1 | i2 | b  | Net| O/p | Thresh")
        #print("neuron display")
        print(*self.inp_list, self.y, self.out, self.threshold, *self.weight_list, sep="  | ")
        # print("Weights used are: ")
        # print(*self.weight_list)

    def get_output(self):
        """Output can be accessed outside the class"""
        return self.out

    def get_weight(self):
        """Returns the current weights"""
        return self.weight_list

    def set_weight(self, new_weights):
        """Change the weights of the neuron"""
        self.weight_list = new_weights


class NeuralNetwork(object):
    """A trainable Perceptron neural network that can forms connections between neurons"""

    def __init__(self):
        """Constructor"""
        self.inp_list = None
        self.target = None
        self.out = list()
        self.l1_num = 0
        self.layer1 = list()
        self.l1op = list()
        self.lr = 1
        ini_w = list()
        ini_w.append([1, 0, 0])
        self.build_net(ini_w)

    def set_input(self, i, t):
        """Updates the inputs and the target"""
        self.inp_list = i
        self.target = t
        cu_wt = list()
        for n in self.layer1:
            cu_wt.append(n.get_weight())
        self.build_net(cu_wt)

    def get_output(self):
        return self.out[0]

    def get_weight(self):
        """Prints the weight of the neural network"""
        cu_wt = list()
        for n in self.layer1:
            cu_wt.append(n.get_weight())
        return cu_wt

    def build_net(self, wts):
        """Build the network architecture"""
        # Number of neurons in 1st layer
        self.l1_num = 1
        self.layer1 = list()
        for i in range(self.l1_num):
            n1 = Neuron(self.inp_list, wts[i], 0.4)
            self.layer1.append(n1)

    def forward_prop(self):
        """Forward propagation through the network"""
        self.l1op = list()
        for n in self.layer1:
            n.forward_prop()
            self.l1op.append(n.get_output())
        self.out = self.l1op

    def update_weight(self):
        """Updates the weights based on the recent forward prop"""
        #print(len(self.out), len(self.target))
        if self.out[0] != self.target[0]:
            #print(self.out[0], self.target[0])
            #print(len(self.layer1))
            for n in self.layer1:
                add_li = [self.lr * self.target[0] * il for il in self.inp_list]
                cur_wt = n.get_weight()
                #print(cur_wt)
                #print(add_li)
                #print(len(cur_wt), len(add_li))
                new_wt = [c+a for c, a in zip(cur_wt, add_li)]
                #print(len(new_wt))
                n.set_weight(new_wt)

    def learn(self):
        """Implements the learning algo"""
        self.forward_prop()
        self.update_weight()

    def display(self):
        """Display data"""
        #print("Layer 1 data")
        for n in self.layer1:
            n.display_stuff()
            #print("NN display")
            print(self.target)


def train_net(nn):
    """Trains the neural network"""
    dataset = list()
    is1 = [1, 1, 1]
    is2 = [1, 0, 1]
    is3 = [0, 1, 1]
    is4 = [0, 0, 1]
    dataset.append(is1)
    dataset.append(is2)
    dataset.append(is3)
    dataset.append(is4)
    t1 = [-1]
    t2 = [1]
    t3 = [1]
    t4 = [-1]
    targets = list()
    targets.append(t1)
    targets.append(t2)
    targets.append(t3)
    targets.append(t4)
    nb_epochs = 50
    cur_epoch = 0
    ini_wt = None
    fin_wt = nn.get_weight()
    while cur_epoch < nb_epochs:
        print("Currently in epoch number: ", cur_epoch + 1)
        c = 0
        for i, t in zip(dataset, targets):
            #print(len(i), len(t))
            #print(i, t)
            nn.set_input(i, t)
            nn.learn()
            nn.display()
            if t == nn.get_output(): c += 1
        ini_wt = fin_wt
        fin_wt = nn.get_weight()
        if fin_wt == ini_wt and c == 4:
            print("Weights no longer updating after epoch number ", cur_epoch + 1, "!!  Training of the XOR function is"
                                                                                   + " complete!!!!")
            break
        cur_epoch += 1
    if cur_epoch == nb_epochs:
        print("Training period has elapsed")
        if fin_wt is not ini_wt:
            print("Training has failed to learn the XOR function after",  nb_epochs, "epochs!")


def main():
    """The main method"""
    print("Building an XOR gate using Perceptron Neural Network that is trained")
    print("Using Perceptron function as the activation function")
    nn = NeuralNetwork()
    ini_wt = nn.get_weight()
    print("Data printed is as follows:-")
    print("i1 | i2 | b  | Net| O/p | Thresh | w1 | w2 | b ")
    print("Target")
    train_net(nn)
    fin_wt = nn.get_weight()
    print("Initial weight was ", *ini_wt)
    print("Final weight is ", *fin_wt)


if __name__ == "__main__":
    main()
