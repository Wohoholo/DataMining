import math
import random

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
