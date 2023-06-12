import pickle
print('_hyper_edges\n\n')
file = open("../dataset/process/train_data_random_mask.pkl", "rb")
data = pickle.load(file)
print(data)
print(len(data))
file.close()

print('='*50)

print('_hyper_labels\n\n')
file = open("../dataset/process/", "rb")
data = pickle.load(file)
print(data)
print(len(data))
file.close()

print('_hyper_labels\n\n')
file = open("../dataset/process/val_data_random_mask.pkl", "rb")
data = pickle.load(file)
print(data)
print(len(data))
file.close()


