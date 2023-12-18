import numpy as np

# change path to data here
pathToData = "./data"

def _retrieve_data():
    training_data_file = open(f"{pathToData}/mnist_train.csv", "r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    test_data_file = open(f"{pathToData}/mnist_test.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    return training_data_list, test_data_list


def mnist_inputs(dataset) -> np.array:
    training_data_list, test_data_list = _retrieve_data()
    training_inputs_list = []
    test_inputs_list = []
    if dataset == "train":
        for i, record in enumerate(training_data_list):
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            training_inputs_list.append(inputs)
        return np.array(training_inputs_list)

    else:
        for i, record in enumerate(test_data_list):
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            test_inputs_list.append(inputs)
        return np.array(test_inputs_list)


def mnist_labels(dataset, output_nodes) -> np.array:
    training_data_list, test_data_list = _retrieve_data()
    training_labels_list = []
    test_labels_list = []
    if dataset == "train":
        for i, record in enumerate(training_data_list):
            all_values = record.split(',')
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            training_labels_list.append(targets)
        return np.array(training_labels_list)
    
    else:
        for i, record in enumerate(test_data_list):
            all_values = record.split(',')
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            test_labels_list.append(targets)
        return np.array(test_labels_list)
