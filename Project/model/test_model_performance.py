import torch
from random import randrange
from torch.utils.data import TensorDataset
from tqdm import tqdm
from unet import UNet
from data_loading import RasterDataLoader

def load_data(path, batch_size):
    RDL = RasterDataLoader(path, in_bands=7)
    X, y = RDL.get_test_training_loaders()
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.long)
    test_data = TensorDataset(X, y)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return test_loader, X, y

def init_model(path):
    model_state = torch.load(path)
    model = UNet(in_chan=7, n_classes=2, depth=3, skip=True, bndry_dropout=True)
    model.load_state_dict(model_state, strict=False)
    return model

def test_model(test_loader, model, device):
    """
    Test the model accuracy using the unseen test set data
    """
    model.eval()

    # Loop over all the validation examples and accumulate the number of correct results in each batch
    total = 0.
    correct = 0.
    correct_none = 0
    correct_new = 0
    total_none = 0
    total_new = 0
    guess_none = 0
    guess_new = 0
    for d, t in test_loader:
        # Send the data to the GPU
        d = d.to(device, dtype=torch.float32)
        t = t.to(device, dtype=torch.float32)

        # Send the data through the model
        outputs = model(d)
        
        # Find which class (label) is most likely for each pixel in each image of the batch
        # predicted = torch.argmax(outputs, 1)
        probs = torch.sigmoid(outputs)
        predicted = torch.argmax(probs, 1)

        # Count the number of correct pixels, and the total number of pixels
        correct += torch.sum(t == predicted)
        total += torch.numel(t)

        # Count the number of correct pixels in each class
        correct_none += torch.sum((predicted == t) * (predicted == 0))
        correct_new += torch.sum((predicted == t) * (predicted == 1))

        # Count the number of predictions for each class (used for overestimation analysis)
        guess_none += torch.sum(predicted == 0)
        guess_new += torch.sum(predicted == 1)

        # Guess the total number of pixels in each class
        total_none += torch.sum(t == 0)
        total_new += torch.sum(t == 1)
        
    # Compute performance metrics
    ratio_guess_none = float(guess_none / total_none)
    ratio_guess_new = float(guess_new / total_new)    
    ratio_correct_none = float(correct_none / total_none)
    ratio_correct_new = float(correct_new / total_new)
    ratio_correct_total = float(correct / total)

    return ratio_guess_none, ratio_guess_new, ratio_correct_none, ratio_correct_new, ratio_correct_total


def run_random_raster(test_loader, model, device):
        for d_total, t_total in test_loader:
            # Get a random raster
            num_rasters = d_total.shape[0]
            idx = randrange(0, num_rasters)

            # Pull the raster from the stack
            d = d_total[idx, ...].reshape(1, 7, 64, 64)
            t = t_total[idx, ...].reshape(1, 64, 64)

            # Send the data to the device
            d = d.to(device, dtype=torch.float32)
            t = t.to(device, dtype=torch.float32)

            # Run the raster through the model
            outputs = model(d)
            probs = torch.sigmoid(outputs)
            predicted = torch.argmax(probs, 1)

            return d[0, ...], t, predicted