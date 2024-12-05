import matplotlib.pyplot as plt


# Veriler
epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
train_loss_sgd1 = [0.53142, 0.55476, 0.50848, 0.34943, 0.15241, 0.22686, 0.21036, 0.12003, 0.16290, 0.16690, 0.10450, 0.17570, 0.17088, 0.11741, 0.10913, 0.06764, 0.10515, 0.13190, 0.10714, 0.07693, 0.08009]
train_loss_sgd2 = [0.53420, 0.48122, 0.29815, 0.25807, 0.19452, 0.12915, 0.14457, 0.14512, 0.08979, 0.06271, 0.07619, 0.07426, 0.05155, 0.07199, 0.06146, 0.04795, 0.06063, 0.05628, 0.05366, 0.03968, 0.05112]
train_loss_sgd3 = [0.53373, 0.37663, 0.20577, 0.16087, 0.09412, 0.09481, 0.07158, 0.07668, 0.04753, 0.04769, 0.02686, 0.03538, 0.04109, 0.04221, 0.04024, 0.04797, 0.03107, 0.04215, 0.03204, 0.03259, 0.02191]
train_loss_sgd4 = [0.53224, 0.21722, 0.07736, 0.04273, 0.03485, 0.03821, 0.02950, 0.02390, 0.02105, 0.02356, 0.02280, 0.02854, 0.02138, 0.02671, 0.01524, 0.02060, 0.02296, 0.02144, 0.02051, 0.01477, 0.02166]
train_loss_sgd5 = [0.52881, 0.08842, 0.02803, 0.02374, 0.02054, 0.01711, 0.01341, 0.01635, 0.01401, 0.01420, 0.01676, 0.01324, 0.01355, 0.01252, 0.00957, 0.01401, 0.01517, 0.01606, 0.01365, 0.01573, 0.01019]


weight_range = 0.001
batch_size_1 = 8
batch_size_2 = 16
batch_size_3 = 32
batch_size_4 = 128
batch_size_5 = 512

# Grafik boyutunu ayarlama
plt.figure(figsize=(12, 12))

# Grafik için veriler
plt.plot(epochs, train_loss_sgd1, label='Size 8', color='green', marker='o')
plt.plot(epochs, train_loss_sgd2, label='Size 16', color='red', marker='o')
plt.plot(epochs, train_loss_sgd3, label='Size 32', color='blue', marker='o')
plt.plot(epochs, train_loss_sgd4, label='Size 128', color='magenta', marker='o')
plt.plot(epochs, train_loss_sgd5, label='Size 512', color='black', marker='o')

# Grafik başlıkları ve etiketler
plt.title(f'SGD Batch Comparison with Weight Initialization in Range [-{weight_range}, {weight_range}]\nLearning rate: 0.001')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Legenda eklemek
plt.legend()

plt.grid(True)
plt.savefig(f'sgd_batch_comparison_weight_range_{weight_range}.png')

# Grafiği göstermek
plt.show()

