import gradio as gr
import torch
import numpy as np

from resnet.resnet import *

def load_cub200_classes():
    """
    This function loads the classes from the classes.txt file and returns a dictionary
    """
    with open("classes.txt", encoding="utf-8") as f:
        classes = f.read().splitlines()

    # convert classes to dictionary separating the lines by the first space
    classes = {int(line.split(" ")[0]) : line.split(" ")[1] for line in classes}

    # return the classes dictionaryg
    return classes

def load_model():
    """
    This function loads the trained model and returns it
    """
    # load the resnet model
    model = Network()
    # load the trained weights
    model = torch.load("cub_classification_resnet.pt", map_location=torch.device('cpu'))
    # Load actual model 
    # model = torch.load("resnet_classification.pt", map_location=torch.device('cpu'))
    # set the model to evaluation mode
    model.eval()
    # return the model
    return model

def predict_image(image):
    """
    This function takes an image as input and returns the class label
    """

    # load the model
    model = load_model()
    # model.eval()
    # load the classes
    classes = load_cub200_classes()

    # convert image to tensor
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    # make prediction
    # soft, prediction = model(tensor)
    prediction_tensor = model(tensor)[1].detach()
    print(f"Prediction Tensor Size: {prediction_tensor.size()}")

    prediction_tensor_max_idx = torch.argmax(prediction_tensor)
    prediction = model(tensor)[1].detach().numpy()[0]


    # convert prediction to probabilities
    print(prediction)
    # print(prediction_tensor)
    # probabilities_tensor = torch.exp(prediction_tensor_max) / torch.sum(torch.exp(prediction_tensor))
    print(f"Probability Tensor Max Index: {prediction_tensor_max_idx}")
    probabilities = np.exp(prediction) / np.sum(np.exp(prediction))
    # get the class with the highest probability
    class_idx = np.argmax(probabilities)
    print(f"Numpy Class Index: {class_idx}")
    # return the class label
    # return "Class: " + classes[class_idx]
    return "Class: " + classes[prediction_tensor_max_idx.item()]

# create a gradio interface
gr.Interface(fn=predict_image, inputs="image", outputs="text").launch()
