using Mocha
using Distributions

# generate inputs
generate_dataset(media,var,tam) = rand(MvNormal(media, var), tam)
# generate outputs
f1(x1, x2) = sin(x1).*sin(x2)./(x1.*x2)

#Parameter Definition for the dataset generation
media_x1=0.0
media_x2=0.0
mean=[media_x1;media_x2]
var_x1=1.0
var_x2=1.0
var=[var_x1 0.0;0.0 var_x2]
#Number of examples (tam)
tam=5000

srand(10)

datasetinput = generate_dataset(mean, var, tam)

datasetoutput = f1(datasetinput[1,:], datasetinput[2,:])

backend = CPUBackend()
init(backend)

data_layer = MemoryDataLayer(name="data", data=Array[datasetinput, datasetoutput], batch_size=100)
ip_layer = InnerProductLayer(name="ip", output_dim=35, bottoms=[:data], tops=[:ip], neuron=Neurons.Tanh())
aggregator = InnerProductLayer(name="aggregator", output_dim=1, tops=[:aggregator], bottoms=[:ip] )
layer_loss = SquareLossLayer(name="loss", bottoms=[:aggregator, :label])

common_layers = [ip_layer, aggregator]

net = Net("MLP", backend, [data_layer, common_layers, layer_loss])

#validation test
input_test = generate_dataset(mean, var, 5000)
output_test = f1(input_test[1,:], input_test[2,:])
data_test = MemoryDataLayer(data = Array[input_test, output_test], batch_size = 100)
accuracy = SquareLossLayer(name="acc", bottoms=[:aggregator, :label])
net_test = Net("test", backend, [data_test, common_layers, accuracy])
test_performance = ValidationPerformance(net_test)

method = SGD() # stochastic gradient descent
params = make_solver_parameters(method, max_iter=1000)
solver = Solver(method, params)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=10)
add_coffee_break(solver, Snapshot("snapshots"), every_n_iter=1000)
add_coffee_break(solver, test_performance, every_n_iter=10)

solve(solver, net)

Mocha.dump_statistics(solver.coffee_lounge, get_layer_state(net, "loss"), true)

destroy(net)
destroy(net_test)
shutdown(backend)
