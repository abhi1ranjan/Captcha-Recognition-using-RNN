import os
import glob
import torch
import numpy as np

import albumentations
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import engine
from model import CaptchaModel


from torch import nn


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin

## this function is used to decode the predictions. preds is a tensor of shape (batch_size, max_len, num_classes)
def decode_predictions(preds, encoder):                                             
    preds = preds.permute(1, 0, 2)                                           ## permute is used to change the order of the dimensions. Here we are changing the order of the first and second dimension. So the shape becomes (max_len, batch_size, num_classes)
    preds = torch.softmax(preds, 2)                                          ## softmax is used to convert the logits to probabilities. Here we are applying softmax on the last dimension. So the shape remains the same
    preds = torch.argmax(preds, 2)                                           ## argmax is used to get the index of the maximum value. Here we are applying argmax on the last dimension. So the shape remains the same
    
    #  detach() is used to detach the tensor from the computational graph. This is done because we are not going to use the gradients in the future. So we can save some memory by detaching the tensor from the computational graph
    preds = preds.detach().cpu().numpy()                                     ## we are converting the tensor to numpy array, and moving it to the CPU, so that we can use it in the CPU.
    cap_preds = []                                                           ## this will contain the decoded predictions of the captcha images.
    for j in range(preds.shape[0]):                                          ## preds.shape[0] is the number of images in the batch, and iterate over those images
        temp = []
        for k in preds[j, :]:                       # going to each values   ## preds[j, :] is the prediction for the jth image. Iterate over the predictions for the jth image
            k = k - 1                               ## we have added a blank class, so we are subtracting 1 from the predictions
            if k == -1:                                       # if the prediction is -1, then it is a blank class, so we are appending a space/unknown character to the temp list
                temp.append("§")
            else:                                             # if the prediction is not -1, then it is a valid character, so we are decoding the prediction and appending it to the temp list
                p = encoder.inverse_transform([k])[0]         # inverse_transform is used to decode the predictions. It takes a list of integers as input, and returns a list of strings as output. Here we are passing a list of length 1, and getting a list of length 1 as output
                temp.append(p)
        tp = "".join(temp).replace("§", "")                   
        cap_preds.append(remove_duplicates(tp))               
    return cap_preds

## we can grab all the png files and create a csv file with the image name and the target. we are not doing that here. But doing that will make things easier 
## making the folds beforehand and saving them in a csv file will make things easier

def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))      ## glob scans through the directory and returns a list of all the files with the extension .png
    
    ## image file is a list. It has paths to all the images like this "/../.../ddsfj.png"
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]    # here slpit("/")[-1] gives the last element of the list after spliting by /, which is the file name. [:-4] removes the last 4 characters, which is .png
    
    ## these targets are strings. They are the names of the images. We need to convert them to integers. We will use label encoding for that
    targets = [[c for c in x] for x in targets_orig]     ## abcde is converted to ['a', 'b', 'c', 'd', 'e'], all the targets are of same size
    
    ## we have list of list targets. Below we convert it into a single list- flat targets
    targets_flat = [c for clist in targets for c in clist]   ## this is a list of all the characters in the targets. This is used to fit the label encoder. 

    lbl_enc = preprocessing.LabelEncoder()   ## label encoder is used to convert the characters to integers
    lbl_enc.fit(targets_flat)                ## fit the label encoder with the flat targets
    targets_enc = [lbl_enc.transform(x) for x in targets]   ## encoded targets. This is a list of list. Each list is the encoded version of the target.
    targets_enc = np.array(targets_enc)
    targets_enc = targets_enc + 1   ## we add 1 to all the targets. This is because we will use 0 for padding/unknown. So we need to shift all the targets by 1

    ## print(targets)
    ## print(np.unique(targets_flat))
    ## print(targets_enc)
    ## print(lbl_enc.classes_)


    ## we will use train_test_split to split the data into train and test. We will use 10% of the data for testing
    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        _,
        test_targets_orig,
    ) = model_selection.train_test_split(
        image_files, targets_enc, targets_orig, test_size=0.1, random_state=42
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))           ## we pass the number of characters to the model
    model.to(config.DEVICE)                                         ## we move the model to the device

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)       ## we use adam optimizer before training. We can use other optimizers as well, all the parameters will be used.
    
    ## we use ReduceLROnPlateau to reduce the learning rate when the loss plateaus. We can use other schedulers as well.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )                                                               
    
    ## training code module 
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)   ## we train the model, and get the loss
        valid_preds, test_loss = engine.eval_fn(model, test_loader)    ## we evaluate the model, and get the loss and predictions
        valid_captcha_preds = []                                       ## we will store the predictions here
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)            ## we decode the predictions, and store them in current_preds
            valid_captcha_preds.extend(current_preds)                  ## we extend the current_preds to the valid_captcha_preds
        combined = list(zip(test_targets_orig, valid_captcha_preds))   ## we combine the original targets and the predicted targets
        print(combined[:10])
        test_dup_rem = [remove_duplicates(c) for c in test_targets_orig]
        accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)
        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} Accuracy={accuracy}"
        )
        scheduler.step(test_loss)


if __name__ == "__main__":
    run_training()
