import os
import sys
import json
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.nn import DataParallel
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import time

from unet import *
from data_loading import *

EPOCHS = 10
EPOCH_STEPS = 1
TRAIN_BATCH_SIZE=20
TEST_BATCH_SIZE=20
LABEL_WEIGHTS=[1, 10, 1000]
DATA_PATH = '/media/anthony/Storage_1/aviation_data/dataset'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
RESULTS = {
    'val_guess_0': [],
    'val_guess_1': [],
    'val_guess_2': [],
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
    'test_guess_0': [],
    'test_guess_1': [],
    'test_guess_2': [],
    'test_correct_total': [],
    'train_time': 0.,
    'epochs': EPOCHS,
    'epoch-steps': EPOCH_STEPS
}

def prep_data_local(dpath) -> tuple:
    """
    This function loads in all data into memory in the form of np arrays. It splits the data into training,
    validation, and test groups based on a 70%, 15%, 15% basis, converts the np arrays to tensors, and finally
    returns train, validate, and test loader objects.

    The dataloaders are torch iterables which handle data batches and iteration.
    """
    # Load all of the raster data into memory using np arrays
    RDL = RasterDataLoader(dpath, in_bands=7)
    X, y = RDL.get_test_training_loaders()

    # split the data into test and training data
    X, X_test_val, y, y_test_val = train_test_split(X, y, test_size=0.7)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5)

    # Convert the np arrays to tensors
    X = torch.from_numpy(X)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)
    y = torch.from_numpy(y)
    y_val = torch.from_numpy(y_val)
    y_test = torch.from_numpy(y_test)

    # Turn the tensors into the appropriate datatype
    X = X.to(torch.float32)
    X_val= X_val.to(torch.float32)
    X_test = X_test.to(torch.float32)
    y = y.to(torch.long)
    y_val = y_val.to(torch.long)
    y_test = y_test.to(torch.long)

    # Create dataset objects
    training_data = TensorDataset(X, y)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)

    # Use the datasets to create train and test loader objects
    train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                               batch_size=TRAIN_BATCH_SIZE,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                              batch_size=TEST_BATCH_SIZE,
                                              shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=TEST_BATCH_SIZE,
                                              shuffle=False)
    return train_loader, val_loader, test_loader


def train_model(train_loader, val_loader, model, epochs) -> nn.Module:
    """
    This function trains the model using machine learning magic
    """
    model.to(DEVICE)

    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    # criterion = torch.nn.CrossEntropyLoss()
    pos_weight = torch.from_numpy(np.array(LABEL_WEIGHTS)).to(torch.float).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=pos_weight)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=1.0e-3)

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

        # Perform model validation after the training step
        if epoch % EPOCH_STEPS == 0:
            model.eval()
            # After each epoch, compute the validation set accuracy
            loss_tracker = 0
            total = 0.
            correct = 0.

            # Loop over all the validation examples and accumulate the number of correct results in each batch
            correct_none = 0
            correct_old = 0
            correct_new = 0
            total_none = 0
            total_old = 0
            total_new = 0
            guess_none = 0
            guess_old = 0
            guess_new = 0
            print('Computing validation accuracy')
            for d, t in val_loader:
                # Send the data to the GPU
                d = d.to(DEVICE, dtype=torch.float32)
                t = t.to(DEVICE, dtype=torch.long)

                # Send the data through the model
                outputs = model(d)

                # Find which class (label) is most likely for each pixel in each image of the batch
                predicted = torch.argmax(outputs, 1)

                # Count the number of correct pixels, and the total number of pixels
                correct += torch.sum(t == predicted)
                total += torch.numel(t)
    
                # Count the number of correct pixels in each class
                correct_none += torch.sum((predicted == t) * (predicted == 0))
                correct_old += torch.sum((predicted == t) * (predicted == 1))
                correct_new += torch.sum((predicted == t) * (predicted == 2))

                # Count the number of predictions for each class (used for overestimation analysis)
                guess_none += torch.sum(predicted == 0)
                guess_old += torch.sum(predicted == 1)
                guess_new += torch.sum(predicted == 2)

                # Guess the total number of pixels in each class
                total_none += torch.sum(t == 0)
                total_old += torch.sum(t == 1)
                total_new += torch.sum(t == 2)

                # Keep track of the loss
                loss = criterion(outputs, t)
                loss_tracker += loss.item()
                
            # Compute performance metrics
            ratio_guess_none = float(guess_none / total_none)
            ratio_guess_old = float(guess_old / total_old)
            ratio_guess_new = float(guess_new / total_new)    
            ratio_correct_none = float(correct_none / total_none)
            ratio_correct_old = float(correct_old / total_old)
            ratio_correct_new = float(correct_new / total_new)
            ratio_correct_total = float(correct / total)

            # Track the validation results
            RESULTS['val_guess_0'].append(ratio_guess_none)
            RESULTS['val_guess_1'].append(ratio_guess_old)
            RESULTS['val_guess_2'].append(ratio_guess_new)
            RESULTS['val_correct_0'].append(ratio_correct_none)
            RESULTS['val_correct_1'].append(ratio_correct_old)
            RESULTS['val_correct_2'].append(ratio_correct_new)
            RESULTS['val_correct_total'].append(ratio_correct_total)
            RESULTS['val_cross_entropy_loss'].append(loss_tracker)

            print(f'\nVALIDATION ACCURACIES: 0: {ratio_correct_none*100:.2f}%, 1: {ratio_correct_old*100:.2f}%, 2: {ratio_correct_new*100:.2f}%')
            print(f'VALIDATION OVERFITTING: 0: {ratio_guess_none:.2f}, 1: {ratio_guess_old:.2f}, 2: {ratio_guess_new:.2f}')
            print(f'TOTAL VALIDATION ACCURACY {ratio_correct_total*100:.4f}%')
            print(f'Cross-Entropy Loss: {loss_tracker:.4f}')
    
    return model

def test_model(test_loader, model) -> tuple:
    """
    Test the model accuracy using the unseen test set data
    """
    model.eval()

    # Loop over all the validation examples and accumulate the number of correct results in each batch
    total = 0.
    correct = 0.
    correct_none = 0
    correct_old = 0
    correct_new = 0
    total_none = 0
    total_old = 0
    total_new = 0
    guess_none = 0
    guess_old = 0
    guess_new = 0
    print('Computing test accuracy')
    for d, t in test_loader:
        # Send the data to the GPU
        d = d.to(DEVICE, dtype=torch.float32)
        t = t.to(DEVICE, dtype=torch.long)

        # Send the data through the model
        outputs = model(d)
        
        # Find which class (label) is most likely for each pixel in each image of the batch
        predicted = torch.argmax(outputs, 1)

        # Count the number of correct pixels, and the total number of pixels
        correct += torch.sum(t == predicted)
        total += torch.numel(t)

        # Count the number of correct pixels in each class
        correct_none += torch.sum((predicted == t) * (predicted == 0))
        correct_old += torch.sum((predicted == t) * (predicted == 1))
        correct_new += torch.sum((predicted == t) * (predicted == 2))

        # Count the number of predictions for each class (used for overestimation analysis)
        guess_none += torch.sum(predicted == 0)
        guess_old += torch.sum(predicted == 1)
        guess_new += torch.sum(predicted == 2)

        # Guess the total number of pixels in each class
        total_none += torch.sum(t == 0)
        total_old += torch.sum(t == 1)
        total_new += torch.sum(t == 2)
        
    # Compute performance metrics
    ratio_guess_none = float(guess_none / total_none)
    ratio_guess_old = float(guess_old / total_old)
    ratio_guess_new = float(guess_new / total_new)    
    ratio_correct_none = float(correct_none / total_none)
    ratio_correct_old = float(correct_old / total_old)
    ratio_correct_new = float(correct_new / total_new)
    ratio_correct_total = float(correct / total)

    # Track the validation results
    RESULTS['test_guess_0'].append(ratio_guess_none)
    RESULTS['test_guess_1'].append(ratio_guess_old)
    RESULTS['test_guess_2'].append(ratio_guess_new)
    RESULTS['test_correct_0'].append(ratio_correct_none)
    RESULTS['test_correct_1'].append(ratio_correct_old)
    RESULTS['test_correct_2'].append(ratio_correct_new)
    RESULTS['test_correct_total'].append(ratio_correct_total)

    print('\n***********************************************************')
    print(f'\nTEST ACCURACIES: 0: {ratio_correct_none*100:.2f}%, 1: {ratio_correct_old*100:.2f}%, 2: {ratio_correct_new*100:.2f}%')
    print(f'\nTEST OVERFITTING: 0: {ratio_guess_none:.2f}, 1: {ratio_guess_old:.2f}, 2: {ratio_guess_new:.2f}')
    print(f'TOTAL TEST ACCURACY {ratio_correct_total*100:.4f}%')


def main(dpath) -> None:
    print('\nWelcome, thank you for training a model today!\n')

    # Time the data loading / training
    start_time = time.time()

    # Initialize the data
    print('\n********************************************\nInitializing Data')
    try:
        train_loader, val_loader, test_loader = prep_data_local(dpath)
    except MemoryError:
        raise MemoryError('Out of memory. wah wah.')
        quit()
        #TODO: Implement custom data loaders

    # Initialize the model
    print('\n********************************************\nInitializing Model')
    model = UNet(in_chan=7, n_classes=3, depth=3)
    
    # Fancy torch magic to magic model go wheeeeeeeeeeeewwwwwwwwweeeeeee fast!
    print('Connecting model to GPU')
    model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        print("Using DataParallel")
        model = DataParallel(model)
    else:
        print('Using One Device')

    # train the model
    print('\n********************************************\nTraining Model')
    model = train_model(train_loader, val_loader, model, EPOCHS)

    # Assess how well the model did
    test_model(test_loader, model)

    # End the clock
    elapsed_time = time.time() - start_time
    print(f'\n***************************************\nTook {elapsed_time:.2f} seconds to train')
    RESULTS['train_time'] = elapsed_time

    # Save the model's state dictionary
    torch.save(model.state_dict(), 'model.nn')

if __name__ == '__main__':
    print('\nBegining launch sequence...')

    # Load in a passed data path
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]

    # Run the training process
    main(DATA_PATH)
    
    # Save the performance data
    with open('results.json', 'w') as fout:
        json.dump(RESULTS, fout)
