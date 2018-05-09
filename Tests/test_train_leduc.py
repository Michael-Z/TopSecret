from NeuralNet.data_stream import DataStream
from NeuralNet.net_builder import NetBuilder
from NeuralNet.trainer import Trainer

# prepare data stream (train and valid)
ds = DataStream(data_path="../Data/TrainingSamples/Leduc/", batch_size=100)
ds.load_train_data(name="train")
ds.load_valid_data(name="valid")

# build network
nb = NetBuilder()
nb.setup_in_out(input_size=73, output_size=72)
nb.setup_hidden_layer(1, hidden_layer_size=50)
network = nb.build_net()
# print(network)

# train the network
trainer = Trainer(network=network, data_stream=ds, learning_rate=0.001)
trainer.train(epoch_count=10)
