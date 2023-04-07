from tqdm import tqdm
import torch
import config

# training function 
def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0      # final loss
    tk0 = tqdm(data_loader, total=len(data_loader))    # progress bar
    for data in tk0:
        for key, value in data.items():          # data is a dictionary, 
            data[key] = value.to(config.DEVICE)   # value is a tensor, put that tensor on the device (GPU)
        optimizer.zero_grad()                     # set the gradient to zero, otherwise it will accumulate, and the optimizer will not work
        _, loss = model(**data)                   # model returns a tuple, the first element is the prediction (we are not using this), the second is the loss
        loss.backward()                           # backpropagation, calculate the gradient, and store it in the tensor
        optimizer.step()                          # update the weights of the model using the optimizer and the gradient stored in the tensor
        fin_loss += loss.item()                   # add the loss to the final loss
    return fin_loss / len(data_loader)            # return the average loss

# evaluation function 
def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0                                  # final loss
    fin_preds = []                                # final predictions
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)
        batch_preds, loss = model(**data)         # model returns a tuple, the first element is the prediction, the second is the loss
        fin_loss += loss.item()                   # add the loss to the final loss
        fin_preds.append(batch_preds)             # add the prediction to the final predictions
    return fin_preds, fin_loss / len(data_loader) # return the final predictions and the average loss
