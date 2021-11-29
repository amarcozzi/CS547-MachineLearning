import os
import sys
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from unet import *
from data_loading import *

DATA_PATH = '/media/anthony/Storage_1/aviation_data/dataset'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def prep_data(dpath) -> tuple:
    # Load all of the raster data into memory using np arrays
    RDL = RasterDataLoader(dpath, in_bands=7)
    X, y = RDL.get_test_training_loaders()

    # split the data into test and training data
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.75)

    # Convert the np arrays to tensors
    X = torch.from_numpy(X)
    X_test = torch.from_numpy(X_test)
    y = torch.from_numpy(y)
    y_test = torch.from_numpy(y_test)

    # Turn the tensors into the appropriate datatype
    X = X.to(torch.float32)
    X_test = X_test.to(torch.float32)
    y = y.to(torch.long)
    y_test = y_test.to(torch.long)

    # Create dataset objects
    training_data = TensorDataset(X, y)
    test_data = TensorDataset(X_test, y_test)

    # Use the datasets to create train and test loader objects
    batch_size = 25
    train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    batch_size = 25
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def main(dpath):
    train_loader, test_loader = prep_data(dpath)
    model = UNet(in_chan=7, n_classes=3, depth=3)
    model.to(DEVICE)

    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    # criterion = torch.nn.CrossEntropyLoss()
    pos_weight = torch.from_numpy(np.array([1e-2, 1e-1, 1])).to(torch.float).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=pos_weight)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=1.0e-3)

    epochs = 5000

    results = np.zeros([4, epochs])
    total_train = 0
    correct_train = 0
    # Loop over the data
    for epoch in range(epochs):
        print(f'\nEPOCH {epoch+1}\n*******************************')

        model.train()

        # Loop over each subset of data
        print('Training Step')
        for d, t in train_loader:
            d = d.to(DEVICE, dtype=torch.float32)
            t = t.to(DEVICE, dtype=torch.long)

            # Zero out the optimizer's gradient buffer
            model.zero_grad()

            # Make a prediction based on the model
            outputs = model(d)

            # Compute the loss
            loss = criterion(outputs, t)

            # Use backpropagation to compute the derivative of the loss with respect to the parameters
            loss.backward()

            # Use the derivative information to update the parameters
            optimizer.step()

            # _, predicted = torch.max(outputs.data, 1)
            # total_train += float(t.size(0))
            # correct_train += float((predicted == t).sum())

        model.eval()
        # After each epoch, compute the test set accuracy
        total = 0.
        correct = 0.
        # Loop over all the test examples and accumulate the number of correct results in each batch
        correct_none = 0
        correct_old = 0
        correct_new = 0
        total_none = 0
        total_old = 0
        total_new = 0
        # model.to('cpu')
        print('Computing test accuracy')
        for d, t in test_loader:
            d = d.to(DEVICE, dtype=torch.float32)
            t = t.to(DEVICE, dtype=torch.long)
            outputs = model(d)
            predicted = torch.argmax(outputs, 1)
            correct += torch.sum(t == predicted)
            total += torch.numel(t)
            # guess_none += int((predicted == 0).sum())
            # guess_old += int((predicted == 1).sum())
            # guess_new += int((predicted == 2).sum())
            correct_none += torch.sum((predicted == t) * (predicted == 0))
            correct_old += torch.sum((predicted == t) * (predicted == 1))
            correct_new += torch.sum((predicted == t) * (predicted == 2))
            total_none += torch.sum(t == 0)
            total_old += torch.sum(t == 1)
            total_new += torch.sum(t == 2)
        ratio_correct_none = float(correct_none / total_none)
        ratio_correct_old = float(correct_old / total_old)
        ratio_correct_new = float(correct_new / total_new)
        result = float(correct / total)
        results[0, epoch] = ratio_correct_none
        results[1, epoch] = ratio_correct_old
        results[2, epoch] = ratio_correct_old
        results[3, epoch] = result
        print(f'\nTEST ACCURACIES: 0: {ratio_correct_none*100}%, 1: {ratio_correct_old*100}%, 2: {ratio_correct_new*100}%')
        print(f'TOTAL TEST ACCURACY {result*100}%')

        # Print the epoch, the training loss, and the test set accuracy.
        # print(epoch, loss.item(), 100. * correct_train / total_train, 100. * correct / total)

    np.save('results.npy', results)
    torch.save(model.state_dict(), 'model.nn')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
    main(DATA_PATH)
