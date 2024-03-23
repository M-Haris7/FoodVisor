from flask import Flask, render_template, request
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
import torchvision.models as models
import math
from torchvision import transforms, datasets, models


app = Flask(__name__)

"""################################## CHECK FOR FILETYPE #####################################################"""

def is_image_file(filename):
    # Get the file extension (e.g., ".jpg",)
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Check if the extension is either ".jpg" or ".png"
    return file_extension in ('.jpg', '.jpeg')

"""################################## PRE PROCESS THE IMAGE #####################################################"""

def process_image(image_path):
    """Process an image path into a PyTorch tensor"""
    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256
    img = img[:3,:,:]

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    # print(img.shape, means.shape)
    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor
    

"""################################## MODEL LOADING from CHECKPOINT ##############################################"""

# Basic details
def model_loading():
    path = 'model/resnet50Mishra.pt'
    # Get the model name
    model_name = os.path.basename(path).split('-')[0]if '-' in os.path.basename(path) else os.path.basename(path).split('.')[0]

    # checkpoint = torch.load(path, map_location = torch.device('cpu'))
    model = torch.load(path, map_location = torch.device('cpu'))


    # if model_name == 'resnet50':
    #     print(model_name)
    #     model = models.resnet50( weights = None )

    #     # model.fc = checkpoint['fc']

    # # Load in the state dict
    # model.load_state_dict(checkpoint['state_dict'])

    # # Model basics
    # model.class_to_idx = checkpoint['class_to_idx']
    # model.idx_to_class = checkpoint['idx_to_class']
    # model.epochs = checkpoint['epochs']

    class_labels = [
        'misi_roti', 'anarsa', 'ledikeni', 'aloo_methi',
       'double_ka_meetha', 'aloo_tikki', 'maach_jhol', 'ariselu',
       'naan', 'mysore_pak', 'misti_doi', 'kachori', 'phirni',
       'litti_chokha', 'gavvalu', 'pithe', 'kuzhi_paniyaram',
       'aloo_gobi', 'chapati', 'kofta', 'kadhi_pakoda', 'chicken_tikka',
       'poornalu', 'chikki', 'pootharekulu', 'modak', 'dal_makhani',
       'dal_tadka', 'paneer_butter_masala', 'doodhpak', 'cham_cham',
       'dum_aloo', 'gajar_ka_halwa', 'bhatura', 'ghevar',
       'dharwad_pedha', 'chak_hao_kheer', 'jalebi', 'chhena_kheeri', 'kadai_paneer',
       'malapua', 'bhindi_masala', 'daal_baati_churma', 'adhirasam', 'imarti',
       'palak_paneer', 'lyangcha', 'navrattan_korma', 'gulab_jamun',
       'kalakand', 'basundi', 'lassi', 'aloo_shimla_mirch',
       'chana_masala', 'biryani', 'unni_appam', 'kakinada_khaja',
       'kajjikaya', 'butter_chicken', 'boondi', 'makki_di_roti_sarson_da_saag', 'karela_bharta', 'poha',
       'chicken_tikka_masala', 'daal_puri', 'aloo_matar',
       'chicken_razala', 'qubani_ka_meetha', 'rabri', 'ras_malai', 'rasgulla',
       'sandesh', 'shankarpali', 'sheer_korma', 'sheera',
       'pootharekulu', 'sohan_halwa', 'sohan_papdi',
       'sutar_feni', 'bandar_laddu'

    ]

   # Create model.class_to_idx list
    model.class_to_idx = [(label, idx) for idx, label in enumerate(class_labels)]

    # Create model.idx_to_class list
    model.idx_to_class = [(idx, label) for idx, label in enumerate(class_labels)]

    return model

"""######################################  PREDICTION FUNCTION  ################################################"""

def predict(image_path, model, topk ):

    """
    Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return
    --------
    Returns
    """

    img_tensor = process_image(image_path)

    img_tensor = img_tensor.reshape(1, 3, 224, 224)

    with torch.no_grad():
        # Set to evaluation
        model.eval()

        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)
        # print(out)
        # print(ps)

        topk, topclass = ps.topk(topk, dim = 1)
        print(topk, topclass)

        print(model.idx_to_class)
        top_classes = [model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes


"""############################### Extract Info from CSV corresponding to the Index  ############"""

def extract(index):

    if ( index < 30 ):
        df = pd.read_csv("info1.csv") 

        # Retrieve column based on index and store as string
        info = df.iloc[index, 2]

        return info
    else :
        return 'Not a valid Index'

"""##########################################   ROUTES    ##################################################"""
# Home Route
@app.route('/')
def index():
    return render_template("index.html")


# when the user hits submit button
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    
    if 'file' not in request.files:
        return 'No file part' 
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected'
    
    if file and is_image_file(file.filename):
    
        #newRepo\static\upload
        img_path = 'static/upload/' + file.filename
        file.save(img_path)
        # print(img_path)

        model = model_loading()

        # Predict Function, takes (imagePath, modelName, number of top precitions to return) as parameters
        img, p, classes = predict(img_path, model, 1)
        result = pd.DataFrame({'p': p}, index = classes)

        img_path = img_path.replace('newRepo/', '../')
        # newRepo\static\upload\neem.jpg
        print(classes[0][0], classes[0][1], p[0])   

        info = extract(classes[0][0])
        # print(info)

       
        return render_template("result.html", img_path = img_path, prediction_name = classes[0][1], confidence_level = p[0]//100, description = info )

    return 'Upload failed. Please check for correct file formats, only jpeg and png are accepted.'

    return render_template("result.html")


"""##################################### MAIN APP CALL #########################################"""
if __name__ == "__main__":
    app.run( debug = True)
