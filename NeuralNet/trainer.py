from torch.legacy import optim
from Settings.arguments import TexasHoldemAgrument
from NeuralNet.masked_huber_loss import MaskedHuberLoss
import time


class Trainer:

	def __init__(self, network, data_stream, learning_rate=None):
		self.network = network
		self.data_stream = data_stream
		self.params, self.grads = self.network.flattenParameters()
		self.criterion = MaskedHuberLoss()
		self.learning_rate = learning_rate or TexasHoldemAgrument.learning_rate
		self.optim_func = optim.adam

	def train(self, epoch_count):
		start_time = time.time()

		for epoch in range(epoch_count):
			self.data_stream.start_epoch()
			loss_sum = 0
			state = {"learningRate": self.learning_rate}
			for i in range(self.data_stream.get_train_batch_count()):
				inputs, targets, mask = self.data_stream.get_train_batch(i)
				# todo learning rate
				_, loss = self.optim_func(lambda x: self.feval(x, inputs, targets, mask), self.params, state)
				loss_sum += loss
			# print("training loss:%d" % loss_sum / self.data_stream.get_train_batch_count())
			print(loss_sum)
			print("epoch %d" % epoch)
		end_time = time.time()
		print("training time cost", end_time - start_time)

		valid_loss_sum = 0
		for i in range(self.data_stream.get_valid_batch_count()):
			inputs, targets, mask = self.data_stream.get_valid_batch(batch_index=i)
			outputs = self.network.forward(inputs)
			loss = self.criterion.forward(outputs, targets, mask)
			valid_loss_sum += loss

		valid_loss = valid_loss_sum / self.data_stream.get_valid_batch_count()
		print("valid loss:", valid_loss)

	# function pass to optim package
	# @param params_new the neural network parameters
	# @param inputs the neural network inputs
	# @param targets the neural network targets
	# @param mask the mask vector used for the loss function
	# @return the masked huber loss on 'inputs' and 'targets'
	# @return the gradient of the loss function
	def feval(self, params_new, inputs, targets, mask):
		# set x to x_new, if different
		# in this simple implementation, it's useless, since x_new always points to x
		if self.params is not params_new:
			self.params.copy_(params_new)

		# zero grads, compute outputs and loss
		self.grads.zero_()
		outputs = self.network.forward(inputs)
		loss = self.criterion.forward(outputs, targets, mask)

		# backward
		dloss_doutput = self.criterion.backward(outputs, targets)
		self.network.backward(inputs, dloss_doutput)

		return loss, self.grads


def main():
	from NeuralNet.net_builder import NetBuilder
	from NeuralNet.data_stream import DataStream
	nb = NetBuilder()
	nb.setup_in_out(input_size=1176, output_size=1176)
	nb.setup_hidden_layer(hidden_layer_count=3, hidden_layer_size=588)
	net = nb.build_net()
	ds = None


if __name__ == "__main__":
	main()

