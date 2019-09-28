inp = InputLayer(1)
hid = HiddenLayer(2, inp)
hid2 = HiddenLayer(2, hid)

'''
# manually add weights and biases
hid.weight_matrix = np.asarray([[2.0], [-4.0]])
hid.bias_vector = np.asarray([2.0, 3.0])

# manually add weights and biases
hid2.weight_matrix = np.asarray([[-1.0, 5.0], [3.0, -25.0]])
hid2.bias_vector = np.asarray([3.0, 2.0])

'''
'''
for q in range(500000):

    in1 = random.randint(0, 1)
    in2 = random.randint(0, 1)

    out1 = bool(in1) != bool(in2)
    # out1 = (in1 or in2)

    i_vector = [in1, in2]

    # inputs the values and feeds forward#
    inp.set_inputs(i_vector)
    hid.forward_propagate_layer()
    hid2.forward_propagate_layer()

    # finds the de/dp of all nodes#
    t_vector = [out1]
    hid2.set_error(t_vector)
    hid2.back_propagate_layer()

    # uses the de/dp of the nodes to change the weights#
    hid2.back_propagate_weights()
    hid.back_propagate_weights()





i_vector = [0, 0]
# inputs the values and feeds forward#
inp.set_inputs(i_vector)
hid.forward_propagate_layer()
hid2.forward_propagate_layer()

print(hid2.activation_vector)

i_vector = [0, 1]
# inputs the values and feeds forward#
inp.set_inputs(i_vector)
hid.forward_propagate_layer()
hid2.forward_propagate_layer()

print(hid2.activation_vector)

i_vector = [1, 0]
# inputs the values and feeds forward#
inp.set_inputs(i_vector)
hid.forward_propagate_layer()
hid2.forward_propagate_layer()

print(hid2.activation_vector)

i_vector = [1, 1]
# inputs the values and feeds forward#
inp.set_inputs(i_vector)
hid.forward_propagate_layer()
hid2.forward_propagate_layer()

print(hid2.activation_vector)

'''

i_vector = [1]
# inputs the values and feeds forward#
inp.set_inputs(i_vector)
hid.forward_propagate_layer()
hid2.forward_propagate_layer()

print("activation vector of hidden layer")
print(hid.activation_vector)

print("activation vector of output layer")
print(hid2.activation_vector)

print("weight matrix of hidden layer")
print(hid.weight_matrix)

print("weight matrix of output layer")
print(hid2.weight_matrix)

print("bias of hidden layer")
print(hid.bias_vector)

print("bias of output layer")
print(hid2.bias_vector)

# finds the de/dp of all nodes#


hid2.set_error([0, 1])
hid2.back_propagate_layer()

# uses the de/dp of the nodes to change the weights#
hid2.back_propagate_weights()
hid.back_propagate_weights()

print("")
print("weight matrix of hidden layer after adjustment")
print(hid.weight_matrix)

print("weight matrix of output layer after adjustment")
print(hid2.weight_matrix)

print("bias of hidden layer after adjustment")
print(hid.bias_vector)

print("bias of output layer after adjustment")
print(hid2.bias_vector)
