import torch
from torch import nn
from torch.nn import functional as F


class CaptchaModel(nn.Module):                                                   # nn.Module is the base class for all neural network modules
    def __init__(self, num_chars):                                               # the constructor, num_chars is the number of characters in the captcha
        super(CaptchaModel, self).__init__()                                     # super() is used to give access to methods and properties of a parent or sibling class
        
        # First layer
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))      # nn.Conv2d is a 2D convolutional layer, 3 is the number of input channels, 128 is the number of output channels, kernel_size is the size of the convolving kernel, padding is the size of the padding
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))                           # nn.MaxPool2d is a 2D max pooling layer - defines no of filters we have, kernel_size is the size of the window to take a max over, 2x2 is the default
        
        # Second layer
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))     # nn.Conv2d is a 2D convolutional layer, 128 is the number of input channels, 64 is the number of output channels, kernel_size is the size of the convolving kernel, padding is the size of the padding
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))                           # nn.MaxPool2d is a 2D max pooling layer - defines no of filters we have, kernel_size is the size of the window to take a max over, 2x2 is the default
        
        # add a linear layer whose input will the output second max pooling layer and output will be 1152 which is multiplication of the height and the channel;
        ## here we had 75 timestamp and for each timestamp we had 1152 features, so we have 75*1152 = 86400 features
        self.linear_1 = nn.Linear(1152, 64)                                     # nn.Linear is a linear layer, 1152 is the input size, 64 is the output size(output size is the number of neurons/features in the layer)
        
        # now we have 75 timestamp and for each timestamp we have 64 features, so we have 75*64 = 4800 features. This is the input to the LSTM layer
        self.drop_1 = nn.Dropout(0.2)                                           # nn.Dropout is a dropout layer, 0.2 is the probability of an element to be zeroed. Dropout doesnot change the size.
        
        # nn.GRU is a Gated Recurrent Unit layer, 64 is the input size, 32 is the hidden size, bidirectional=True means that we have a forward and a backward LSTM, num_layers=2 means that we have 2 LSTM layers, dropout=0.25 means that we have a dropout of 0.25, batch_first=True means that the input and output tensors are provided as (batch, seq, feature)
        self.lstm = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)     
        # nn.Linear is a linear layer, 64 is the input size, num_chars + 1 is the output size(output size is the number of neurons/features in the layer), +1 because we have a blank space/unknown character.
        self.output = nn.Linear(64, num_chars + 1)                            

    def forward(self, images, targets=None):           # forward pass, images is the input image, targets is the target, should take the same name whatever is coming from the dataloader
        bs, _, _, _ = images.size()                    # bs is the batch size, _ is the height, _ is the width, _ is the channel
        x = F.relu(self.conv_1(images))                # F.relu is the rectified linear unit function, self.conv_1(images) is the output of the first convolutional layer
        # print(x.size())                                # torch.Size([8, 128, 75, 300]), always print size to check if the dimensions are correct in the neural network
        x = self.pool_1(x)                             # Perform max pooling over the features in the input tensor
        # print(x.size())                                # torch.Size([8, 128, 37, 150]), always print size to check if the dimensions are correct in the neural network
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)                          # 1, 64, 18, 75 is the size of the output after the 2nd max pooling layer

        """_summary_
        After 1st convulation layer: size remains same, 1 is batch size, 128 filters, 75 height, 300 width. 
        After 1st max pooling layer: size reduces to 1/2, 1 is batch size, 128 filters, 37 height, 150 width.
        After 2nd convulation layer: size remains same, 1 is batch size, 64 filters, 37 height, 150 width.
        After 2nd max pooling layer: size reduces to 1/2, 1 is batch size, 64 filters, 18 height, 75 width.
        Returns:
            _type_: _description_
        """

        x = x.permute(0, 3, 1, 2)        #1,75,64,18   # permute the dimensions of an array, batch size index remain same, width index goes to 1st position, channel index goes to 2nd position, height index goes to 3rd position
        ## we are doing this because we want to pass the width as the sequence length to the LSTM layer when we apply our RNN model
        # print(x.size())
        
        x = x.view(bs, x.size(1), -1)                 # 1, 75, 1152   # view is used to reshape the tensor, batch size (1) index remain same, width index remain same, height and channel index are multiplied and reshaped to 1 dimension
        # print(x.size())
        x = F.relu(self.linear_1(x))                  # 1, 75, 64     # output of the linear layer
        # print(x.size())
        x = self.drop_1(x)                            
        # print(x.size())
        x, _ = self.lstm(x)                           # 1, 75, 64     # output of the LSTM layer
        # print(x.size())
        x = self.output(x)                            # 1, 75, 20     # output of the linear layer
        # print(x.size())
        # Have to permute again to get the correct dimensions for the CTC loss
        x = x.permute(1, 0, 2)                        # batch size will go to middle position, timestamp will go to 1st position, features/values will go to 3rd position  


        # For returning the loss.  - used CTC loss
        if targets is not None:
            log_probs = F.log_softmax(x, 2)               # CTC loss requires the output to be log softmax, 2 is the dimension along which the softmax is applied, x is the output of the linear layer
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None                                     # return the output and the loss


if __name__ == "__main__":
    cm = CaptchaModel(19)                                # 19 is the number of characters in the captcha  
    img = torch.rand((1, 3, 75, 300))                    # 1 is the batch size, 3 is the number of channels, 75 is the height, 300 is the width, we choose this in the config file
    x, _ = cm(img, torch.rand((1, 5)))                   # target = torch.rand((1, 5)), 1 is the batch size, 5 is the length of the captcha.
