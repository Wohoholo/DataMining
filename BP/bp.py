import random
from matplotlib import pyplot as plt
import math
# from Neuron import Neuron, NeuronLayer
from util import createData

"""
abbreviation:
pd_ : partial derivative
d_: derivative
_wrt_: with respect to
w_ho: index of weights from hidden to output
w_ih: index of weights from input to hidden
"""

"""
y = (X1-1)**4 + 2*X2**2

first time: 2019/12/10 8:23
---------------------------
update: 2019/12/10/23:44
add: can add more hidden layers in NeuralNetwork
updated: forward function have updated
---------------------------
updated:2019/12/11/10:13
build a network 2-4-4-2-1
complete almost function
but errors is stable
"""

# import math
# import random

class Neuron:

    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        # print(inputs)
        self.output = self.squash(self.calculate_total_net_input())
        # print(self.output)
        return self.output

    def calculate_total_net_input(self):
        total = 0
        # print(self.weights)
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]

        return total +self.bias

    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # pd_error_wrt_total_net_input :
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_output_wrt_total_net_input()

    # calculate error
    def calculate_error(self, target_output):
        return 0.5 * (target_output-self.output)**2

    # pd_error_wrt_output
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output-self.output)

    # pd_output_wrt_total_net_input
    def calculate_pd_output_wrt_total_net_input(self):
        return self.output*(1-self.output)

    # pd_total_net_input_wrt_weight
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]



class NeuronLayer:

    def __init__(self, num_neurons, bias):

        self.bias = bias if bias else random.random()

        self.neurons = []

        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons: ', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron ', n)
            for w in range(len(self.neurons[n].weights)):
                print(' Weight:', self.neurons[n].weights[w])
            print(' Bias:', self.bias)

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class NeuralNetwork:
    LEARNING_RATE = 0.5


    def __init__(self, num_inputs, num_hidden_layers, num_hidden, num_outputs, hidden_layer_weights=None, hidden_layer_bias=None, output_layer_weights=None, output_layer_bias=None):
        """

        :param num_inputs: int
        :param num_hidden_layers: int
        :param num_hidden: list[int]
        :param num_outputs: int
        :param hidden_layer_weights: list[list[int]]
        :param hidden_layer_bias: list[int]
        :param output_layer_weights: list[int]
        :param output_layer_bias: int
        """
        self.num_inputs = num_inputs

        self.hidden_layers = [NeuronLayer(num_hidden[i], hidden_layer_bias[i]) for i in range(num_hidden_layers)]
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_to_output_layer(output_layer_weights)


    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):

        weight_num = 0
        for h in range(len(self.hidden_layers[0].neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layers[0].neurons[h].weights.append(random.random())
                else:
                    self.hidden_layers[0].neurons[h].weights.append(hidden_layer_weights[0][weight_num])
                weight_num += 1

        weight_num = 0
        for h in range(len(self.hidden_layers[1].neurons)):
            for i in range(len(self.hidden_layers[0].neurons)):
                if not hidden_layer_weights:
                    self.hidden_layers[1].neurons[h].weights.append(random.random())
                else:
                    self.hidden_layers[1].neurons[h].weights.append(hidden_layer_weights[1][weight_num])
                weight_num += 1

        weight_num = 0
        for h in range(len(self.hidden_layers[2].neurons)):
            for i in range(len(self.hidden_layers[1].neurons)):
                if not hidden_layer_weights:
                    self.hidden_layers[2].neurons[h].weights.append(random.random())
                else:
                    self.hidden_layers[2].neurons[h].weights.append(hidden_layer_weights[2][weight_num])
                weight_num += 1


    def init_weights_from_hidden_layer_to_output_layer(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layers[-1].neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        for i in range(len(self.hidden_layers)):
            print('------')
            print('Hidden Layer {}'.format(i))
            self.hidden_layers[i].inspect()

        print('------')
        self.output_layer.inspect()
        print('------')

    def forward(self, inputs):

        hidden_layer_outputs = self.hidden_layers[0].forward(inputs)

        for i in range(1, len(self.hidden_layers)):
            hidden_layer_outputs = self.hidden_layers[i].forward(hidden_layer_outputs)

        return self.output_layer.forward(hidden_layer_outputs)

    def train(self, training_inputs, training_outputs):

        self.forward(training_inputs)

        ## calculate delta Wij
        # pd_errors_wrt_output_neuron_total_net_input
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # pd_errors_wrt_hidden_neuron_2_total_net_input
        pd_errors_wrt_hidden_neuron_2_total_net_input = [0] * len(self.hidden_layers[2].neurons)
        for h in range(len(self.hidden_layers[2].neurons)):
            d_error_wrt_hidden_neuron_2_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_2_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]
            pd_errors_wrt_hidden_neuron_2_total_net_input[h] = d_error_wrt_hidden_neuron_2_output * self.hidden_layers[2].neurons[h].calculate_pd_output_wrt_total_net_input()

        # pd_errors_wrt_hidden_neuron_1_total_net_input
        pd_errors_wrt_hidden_neuron_1_total_net_input = [0] * len(self.hidden_layers[1].neurons)
        for h_1 in range(len(self.hidden_layers[1].neurons)):
            d_error_wrt_hidden_neuron_1_output = 0
            for h_2 in range(len(self.hidden_layers[2].neurons)):
                d_error_wrt_hidden_neuron_1_output += pd_errors_wrt_hidden_neuron_2_total_net_input[h_2] * self.hidden_layers[2].neurons[h_2].weights[h_1]
            pd_errors_wrt_hidden_neuron_1_total_net_input[h_1] = d_error_wrt_hidden_neuron_1_output * self.hidden_layers[1].neurons[h_1].calculate_pd_output_wrt_total_net_input()

        # pd_errors_wrt_hidden_neuron_0_total_net_input
        pd_errors_wrt_hidden_neuron_0_total_net_input = [0] * len(self.hidden_layers[0].neurons)
        for h_0 in range(len(self.hidden_layers[0].neurons)):
            d_error_wrt_hidden_neuron_0_output = 0
            for h_1 in range(len(self.hidden_layers[1].neurons)):
                d_error_wrt_hidden_neuron_0_output += pd_errors_wrt_hidden_neuron_1_total_net_input[h_1] * self.hidden_layers[1].neurons[h_1].weights[h_0]
            pd_errors_wrt_hidden_neuron_0_total_net_input[h_0] = d_error_wrt_hidden_neuron_0_output * self.hidden_layers[0].neurons[h_0].calculate_pd_output_wrt_total_net_input()

        ## update weights
        # output layer
        for o in range(len(self.output_layer.neurons)):

            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # hidden layer 2
        for h in range(len(self.hidden_layers[2].neurons)):
            for w_hh in range(len(self.hidden_layers[2].neurons[h].weights)):

                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_2_total_net_input[h] * self.hidden_layers[2].neurons[h].calculate_pd_total_net_input_wrt_weight(w_hh)
                self.hidden_layers[2].neurons[h].weights[w_hh] -= self.LEARNING_RATE * pd_error_wrt_weight

        # hidden layer 1
        for h in range(len(self.hidden_layers[1].neurons)):
            for w_hh in range(len(self.hidden_layers[1].neurons[h].weights)):

                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_0_total_net_input[h] * self.hidden_layers[1].neurons[h].calculate_pd_total_net_input_wrt_weight(w_hh)
                self.hidden_layers[1].neurons[h].weights[w_hh] -= self.LEARNING_RATE * pd_error_wrt_weight

        # hidden layer 0
        for h in range(len(self.hidden_layers[0].neurons)):
            for w_ih in range(len(self.hidden_layers[0].neurons[h].weights)):

                pd_errors_wrt_weight = pd_errors_wrt_hidden_neuron_0_total_net_input[h] * self.hidden_layers[0].neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)
                self.hidden_layers[0].neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_errors_wrt_weight


    def calculate_total_error(self, training_sets):

        total_error = 0
        for i in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[i]
            self.forward(training_inputs)

            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])



        return total_error

    def save(self, file='NetWork.txt'):

        with open(file, 'w') as f:
            for i in range(len(self.hidden_layers)):
                for n in range(len(self.hidden_layers[i].neurons)):
                    f.write(str(self.hidden_layers[i].neurons[n].weights)+'\n')
                f.write(str(self.hidden_layers[i].bias)+'\n')

            for o in range(len(self.output_layer.neurons)):
                f.write(str(self.output_layer.neurons[o].weights)+'\n')
            f.write(str(self.output_layer.bias)+'\n')

if __name__ == '__main__':



    epoch = 5
    plt.figure()
    weights = [[random.random() for i in range(8)],
               [random.random() for i in range(16)],
               [random.random() for i in range(8)]]

    nn = NeuralNetwork(2, 3, [4, 4, 2], 1, hidden_layer_weights=weights, hidden_layer_bias=[0.35, 0.2, 0.4], output_layer_weights=[0.4, 0.45], output_layer_bias=0.6)

    # nn.inspect()
    data = createData()

    x = []
    y = []

    f = open('log.txt', 'w')
    for j in range(epoch):
        for i in range(10000):
            for d in data:
                nn.train(d[:2], d[2:])
                # print(d)

            if i % 100 == 0:
                error = round(nn.calculate_total_error([[[0.7, 0.95], [1.8131]]]), 4)
                f.write('{} times. errors: {}\n'.format(i, error))
                x.append(i)
                y.append(error)


    f.close()
    nn.save()
    x1 = 0.5
    x2 = 0.5
    Y = (x1 - 1) ** 4 + 2 * (x2 ** 2)
    print('Y:', Y)
    print(nn.forward([x1, x2]))
    plt.plot(x, y)
    plt.show()
