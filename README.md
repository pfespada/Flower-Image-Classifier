# Flower-Image-Classifier

This projec is about a image classifier using deep learning to recognize different species of flowers.

The proyect is broken down into multiple steps:

- Load and preprocess the image dataset

- Train the image classifier on your dataset

- Use the trained classifier to predict image content

I used a pre-trained deep network (convolutional) VGG16

This project was used for the Data Scientis Nanodegree (UDACITY)


# Files
This project includes the following files
- Image Classifier Project.ipynb: Jupyter Notebook where the analysis is done
- Data files: listings.csv, reviews_m.csv, calendar_m.csv, neighbothoods_m in zip format.
- Helper.py: files with created function to be used in Image Classifier Project.ipynb
- Predict.py: file with funtion to make prediction and load the checkpoint (classifier)
- Helper.py : function to process the images (process_image) before enter the model and funtion (loaders) to create the dataloaders 
- train.py: script yo train the network
- cat_to_name.json: json file with the name of the image.

# Notes

This image classifier can be used for any picture however it has to be trained accordingly. The folder structure of the pictures are as follows

\Train\type1\pic1, pc2, pic3....
\Train\type2\pic1, pc2, pic3....
\Train\type3\pic1, pc2, pic3....
.
.
\Test\type1\pic1, pc2, pic3....
\Test\type2\pic1, pc2, pic3....
\Test\ptype3\pic1, pc2, pic3....
.
.
\valid\type1\pic1, pc2, pic3....
\valid\type2\pic1, pc2, pic3....
\valid\type3\pic1, pc2, pic3....


# Instalation

For using this notwbook you must have the following python liberies installed

-numpy (https://docs.scipy.org/doc/numpy/user/install.html)

-pandas (https://pandas.pydata.org/pandas-docs/stable/install.html)

-matplotlib (https://matplotlib.org/users/installing.html)

-seaborn (https://seaborn.pydata.org/installing.html)

-torch (https://pytorch.org)

-PIL (https://pillow.readthedocs.io/en/stable/installation.html)

# Acknowledgements
I want to thank UDACITY for the oportunity of doing this project in the Data Scientist ND

# Contribution 
Contributions are welcome!! In case you are interesting to contribute please contact me.
