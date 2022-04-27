# BARK-ode 🐶
![Image](https://github.com/Ashish-Abraham/Bark-ode/blob/main/Images/Explainer-Best-Dog-Breed-For-Me.jpg)

Dogs are not our whole life, but they make our lives whole. The 20,000 years of friendship made them destined to be called “man’s best friend”. As humans became more sophisticated, so did their dogs. Eventually, there emerged specific breeds of dogs, custom-bred to suit the breeders’ local needs and circumstances with over 340 of them known. 
Here is an ML powered tool that will help you to identify 120 of them. The model architecture is a ResNet18 that was initially trained on the ImageNet Dataset. Utilized transfer learning to fine tune the network to identify 120 breeds of dogs with an accuracy of about 90%.<br>The scripts also contain code to perform fine tuning or feature extraction of 5 other state-of-the-art CNN networks available. 
## Neural Network
* ResNet-18
* Framework : PyTorch (1.11.0)
* Validation Score : 0.902
* Dataset : http://vision.stanford.edu/aditya86/ImageNetDogs/ <br><br>

## Model Training
The code for training can be found in src/GetModel.py. The other architectures available are AlexNet, DenseNet, VGG, InceptionV3 and Squeezenet. In order to change architectures, change **model_name** in the below section of **_setup_model()**.
```
model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
```
Put **feature_extract= False** to perform fine tuning and set to **True** for feature extraction. <br> For training InceptionV3, set **is_inception=True** in the below function.
```
train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False)
```
 
## How to use:
* Clone repository to local machine and open directory src.<br>
```
 streamlit run app.py
```
* [WebApp:](https://share.streamlit.io/ashish-abraham/bark-ode/main/src/app.py)<br><br>
![Image](https://github.com/Ashish-Abraham/Bark-ode/blob/main/Images/barkodeui1.jpeg)
![Image](https://github.com/Ashish-Abraham/Bark-ode/blob/main/Images/barkodeui2.jpeg)<br>
* See Wikipedia page of the breed by clicking on the names in case you wish to learn more🤗 .

## To Do:
* Add more features to web-app using Dockerfile included
* Release it as a functional api


