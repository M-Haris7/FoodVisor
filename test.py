import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image

# Assuming you've already defined your model architecture and trained it
# Load the trained model
model = torch.load('model/resnet50Mishra.pt',  map_location = torch.device('cpu'))  # Load the saved model

# Set the model to evaluation mode
model.eval()

# Define transformations for your input data
# Example: Assuming input size of 224x224 pixels
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

# Load your input image
image_path = 'E:\aloo_gobi\0b8b93a61c.jpg'
image = Image.open(image_path)

# Apply transformations to the image
input_image = transform(image).unsqueeze(0)  # Add batch dimension

# If CUDA is available, move the input to the GPU
# if cuda:
#     input_image = input_image.cuda()

# Perform prediction
with torch.no_grad():
    output = model(input_image)

# Get the predicted class
_, predicted_class = torch.max(output, 1)

# Print the predicted class
print("Predicted class:", predicted_class.item())
