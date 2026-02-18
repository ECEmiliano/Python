import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data", #Creates/Checks the data folder. Writes the contents of the dataset.
    train=True, #Checks if the dataset is for training or testing. If True, creates/Checks the training data. If False, creates/Checks the test data.
    download=True, 
    transform=ToTensor(), #Converts the PIL Image to a tensor.
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size) #While the dataaset is static, we can use dataloader to utilise the dataset (Create batches, shuffle data etc.)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}") #N is the batch size, C is the number of channels (1 for grayscale), H and W are the height and width of the images (28x28).
    print(f"Shape of y: {y.shape} {y.dtype}") #y is a 1D tensor of labels corresponding to the images in X
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" #Checks if an accelerator (GPU) is available. If it is, it returns the type of accelerator (e.g., "cuda" for NVIDIA GPUs, "mps" for Apple Silicon). If no accelerator is available, it defaults to "cpu".
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module): 
    def __init__(self): 
        super().__init__() #Calls __init__ of the parent class (nn.Module) to properly initialize the model
        self.flatten = nn.Flatten() #Flattens the input tensor from (N, C, H, W) to (N, C*H*W). In this case, it will flatten the 28x28 images into a 784-dimensional vector. Easier to work with
        self.linear_relu_stack = nn.Sequential(  
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),        #Im lost 
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() #Computes the gradient for each parameter and holds the value
        optimizer.step() #Updates the value of each parameter based on the current gradient (stored in .grad) and the learning rate
        optimizer.zero_grad() #Resets the gradient values because the gradient is cummulative

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

classes = [ #The classes in the FashionMNIST dataset
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval() #This block is to make the machine guess a singular item
x, y = test_data[238][0], test_data[238][1]
#x = x.unsqueeze(0)  Add batch dimension. Works without it because the current images are grayscale (1 channel).
with torch.no_grad():
    x = x.to(device) #We work on the GPU, the data must be transferred to the correct device. Non-necessary for y since we dont compute any mathematical operations with it.
    pred = model(x)
    #print(f"Input tensor shape: {x}").  Prints out the whole tensor matrix. Just wanted to see it.
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')