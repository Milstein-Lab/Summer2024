import spiral_neuralnet as spiral
from spiral import *

def main():
    # Finding optimal learning rate
	spiral.set_seed(SEED)
	g_seed.manual_seed(SEED)
	net1 = Net(nn.ReLU, X_train.shape[1], [128, 32], K).to(DEVICE)
	criterion = "MSELoss"
	num_epochs = 2

	loss_history = []
	learning_rates = np.arange(0.15, 0.25 + 0.01, 0.01)
	for i in learning_rates:
		optimizer = optim.SGD(net1.parameters(), lr=i)
		loss = net1.train_model(criterion, optimizer, train_loader, num_epochs=num_epochs, verbose=True, device=DEVICE)
		loss_history.append(loss)

	plt.figure()
	plt.plot(learning_rates, loss_history)

if __name__ == "__main__":
    main()