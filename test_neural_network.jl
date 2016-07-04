using Mocha
using Distributions
using PyCall

@pyimport mpl_toolkits.mplot3d as mplot3d
@pyimport matplotlib.pyplot as plt

srand(500)

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

datasetinput=generate_dataset(mean, var, tam)
datasetoutput=f1(datasetinput[1,:], datasetinput[2,:])

#println(datasetinput)
#println(datasetoutput)

backend = CPUBackend()
init(backend)

data_layer = MemoryDataLayer(name="data", data=Array[datasetinput, datasetoutput], batch_size=5000, tops=[:data,:label])
ip_layer = InnerProductLayer(name="ip", output_dim=35, bottoms=[:data], tops=[:ip], neuron=Neurons.Tanh())
aggregator = InnerProductLayer(name="aggregator", output_dim=1, tops=[:aggregator], bottoms=[:ip] )

common_layers = [ip_layer, aggregator]

net = Net("MLP", backend, [data_layer, common_layers])

load_snapshot(net, "snapshots/snapshot-001000.jld")

forward(net)
#println(net.output_blobs[:aggregator].data)
println(sum((net.output_blobs[:aggregator].data-datasetoutput).^2)/(2*size(datasetoutput, 2)))

#println(abs(100 .- ((net.output_blobs[:aggregator].data.*100)./datasetoutput)))
destroy(net)

# plot 
fig = plt.figure()
ax = mplot3d.Axes3D(fig)
ax[:set_xlabel]("X Label")
ax[:set_ylabel]("Y Label") 
ax[:set_zlabel]("Z Label")
# plot original data
ax[:scatter](datasetinput[1,:], datasetinput[2,:], datasetoutput, c="r", marker="o")
# plot predictions
ax[:scatter](datasetinput[1,:], datasetinput[2,:], net.output_blobs[:aggregator].data, c="b", marker="^")
plt.show()

shutdown(backend)
