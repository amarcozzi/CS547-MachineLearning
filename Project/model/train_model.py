import os
import sys
import json
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import time

from unet import *
from data_loading import *

DATA_PATH = '/media/anthony/Storage_1/aviation_data/dataset'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
RESULTS = {
    'val_correct_0': [],
    'val_correct_1': [],
    'val_correct_2': [],
    'val_correct_total': [],
    'val_cross_entropy_loss': [],
    'train_correct_0': [],
    'train_correct_1': [],
    'train_correct_2': [],
    'train_correct_total': [],
    'train_cross_entropy_loss': [],
    'test_correct_0': [],
    'test_correct_1': [],
    'test_correct_2': [],
    'test_correct_total': [],
    'train_time': 0.
}

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
    batch_size = 50
    train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    batch_size = 50
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
        loss_tracker = 0
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
 
            correct_none += torch.sum((predicted == t) * (predicted == 0))
            correct_old += torch.sum((predicted == t) * (predicted == 1))
            correct_new += torch.sum((predicted == t) * (predicted == 2))
            total_none += torch.sum(t == 0)
            total_old += torch.sum(t == 1)
            total_new += torch.sum(t == 2)

            loss = criterion(outputs, t)
            loss_tracker += loss.item()
            
        ratio_correct_none = float(correct_none / total_none)
        ratio_correct_old = float(correct_old / total_old)
        ratio_correct_new = float(correct_new / total_new)
        ratio_correct_total = float(correct / total)

        RESULTS['val_correct_0'].append(ratio_correct_none)
        RESULTS['val_correct_1'].append(ratio_correct_old)
        RESULTS['val_correct_2'].append(ratio_correct_new)
        RESULTS['val_correct_total'].append(ratio_correct_total)
        RESULTS['val_cross_entropy_loss'].append(loss_tracker)

        print(f'\nTEST ACCURACIES: 0: {ratio_correct_none*100:.4f}%, 1: {ratio_correct_old*100:.4f}%, 2: {ratio_correct_new*100:.4f}%')
        print(f'TOTAL TEST ACCURACY {ratio_correct_total*100}%')
        print(f'Cross-Entropy Loss: {loss_tracker}')

    # Save the model's state dictionary
    torch.save(model.state_dict(), 'model.nn')

if __name__ == '__main__':
    start_time = time.time()
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
    main(DATA_PATH)
    elapsed_time = time.time() - start_time
    print(f'\n***************************************\nTook {elapsed_time:.2f} seconds to train')
    RESULTS['train_time'] = elapsed_time
    with open('results.json', 'w') as fout:
        json.dump(RESULTS, fout)
