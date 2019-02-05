# Pet Adoption Prediction Model

The goal of this project was to predict the adoption rate for animals at a shelter. Data from Petfinder.my (an animal shelter in Malaysia) was used to build this model.

There were various sources of data, such as a CSV on the individual animal's health, sterilisation, vaccinations, age, colour, breed etc.
I also had metadata on images of the pets that were passed through Google's Vision API, and the descriptions that were used in ads for the pets that were passed through Google's Natural Language API.

The data was quite dirty, as some of the descriptions weren't in English, and the quantity of some animals were 20 per advert. More information on this can be found in my EDA notebook (DataExploration).

Some data processing and feature engineering was done in the preprocess.py script. Note that since most of the features I used were categorical features, I one-hot encoded these features. Hence, I mostly worked with sparce matrices. However, in the embedding_layer_model.py script I built a neural net where I used the cleaned but raw features to build embedding layers.

I built various models, such as the embedding layer model I mentioned before, as well as a one-vs-rest classifier, a simple neural net, a stacking model and a voting classifier. 

The models aren't performing as well as I wanted, but I suspect it has a lot to do with the data that can only be cleaned up to a certain point. Since I had so little data, I didn't want to remove any data for the purpose of cleaning so I tried my best with what I had.
