# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                       Sentiment Analysis using BERT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Workflow:
# ---------
# I.   Download and Install BERT from HF Transformers on HuggingFace
        # Source: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
# II.  Perform Sentiment scoring using BERT
# III. Scrape reviews from Yelp and Score

# 1. Install and Import Dependencies
# ---------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# AutoTokenizer: load the tokenizer that allows to parse through a string 
#       and convert it to a sequence of numbers that can be used as input for the NLP model
# AutoModelForSequenceClassification: provides the transformers structure for the NLP model
import torch
import requests # for scraping reviews from Yelp
from bs4 import BeautifulSoup # for scraping reviews from Yelp
import re
import pandas as pd
import numpy as np
from helper import *


# 2. Instantiate Model
# --------------------
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# 3. Encode and Calculate Sentiment
# ---------------------------------

# convert a string into a sequence of numbers that can be used as input for the NLP model
tokens = tokenizer.encode("Naja, war ganz ok", return_tensors="pt")

# Note: to convert the tokens back into strings, use "tokenizer.decode(tokens[0])"

# plug the tokens into the model and get the output
result = model(tokens)

# get the score of the sentiment as numbers between 1 and 5, 
# with 1 being the most negative and 5 being the most positive
sentiment = int(torch.argmax(result.logits)) + 1 # torch.argmax(result.logits) gives the position of the highest value in the tensor

# 4. Collect Reviews
# ------------------

# scraping reviews of a specific restaurant from yelp:

r = requests.get("https://www.yelp.com/biz/tandoori-night-fresno") # get the HTML of the page
soup = BeautifulSoup(r.text, "html.parser") # parse the HTML
regex = re.compile('.*comment.*')   # define a regex that matches the class name of the HTML elements that contain the reviews
results = soup.find_all("p", {"class": regex}) # find all HTML elements that match the regex
reviews = [result.text for result in results] # extract the text from the HTML elements

# Google reviews can be scraped in a similar way
# r = requests.get("https://www.google.com/maps/place/Tandoori+N%C3%A4chte,+Tandoori+Nights/@52.4876779,13.2981374,14z/data=!4m7!3m6!1s0x47a851472dc72859:0x66f5eb3aa34e833a!8m2!3d52.4962467!4d13.2858839!9m1!1b1") # get the HTML of the page
# soup = BeautifulSoup(r.text, "html.parser") # parse the HTML
# regex = re.compile('.*ODSEW-ShBeI NIyLF-haAclf gm2-body-2.*')   # define a regex that matches the class name of the HTML elements that contain the reviews
# results = soup.find_all("div", {"class_": regex}) # find all HTML elements that match the regex
# reviews = [result.text for result in results] # extract the text from the HTML elements
# reviews

# 5. Load Reviews into DataFrame and Score
# ---------------------------------------

df = pd.DataFrame(np.array(reviews), columns=["review"]) # load the reviews into a dataframe, with the column name "review"

def sentiment_score(review):
    """Calculate the sentiment score of a review"""
    tokens = tokenizer.encode(review, return_tensors="pt")
    result = model(tokens)
    sentiment = int(torch.argmax(result.logits)) + 1
    return sentiment

# sentiment_score(df['review'].iloc[0]) # test the function; returns the sentiment score of the first review

df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512])) # apply the function to all reviews and 
# add the sentiment score to the dataframe; Note that the model can only handle 512 tokens at a time, 
# so we slice the review to the first 512 characters