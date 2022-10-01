# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                       Sentiment Analysis using BERT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Workflow:
# ---------
# I.   Download and Install BERT from HF Transformers on HuggingFace
        # Source: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
# II.  Perform Sentiment scoring using BERT
# III. Scrape reviews from Yelp and Score

# 1. Import Dependencies
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
import math
import matplotlib.pyplot as plt
# from functions import *


# 2. Instantiate Model
# --------------------
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# 3. Encode and Calculate Sentiment (example)
# -------------------------------------------

# convert a string into a sequence of numbers that can be used as input for the NLP model
tokens = tokenizer.encode("I love this restaurant", return_tensors="pt")

# Note: to convert the tokens back into strings, use "tokenizer.decode(tokens[0])"

# plug the tokens into the model and get the output
result = model(tokens)

# get the score of the sentiment as numbers between 1 and 5, 
# with 1 being the most negative and 5 being the most positive
sentiment = int(torch.argmax(result.logits)) + 1 # torch.argmax(result.logits) gives the position of the highest value in the tensor

# 4. Collect Reviews
# ------------------

url_home = "https://www.yelp.com/biz/tandoori-night-fresno"
# url_home = "https://www.yelp.de/biz/anand-berlin"
r = requests.get(url_home, verify=False) # get the HTML of the page
soup = BeautifulSoup(r.text, "html.parser") # parse the HTML

# find the total number of reviews:
regex_count = re.compile('.*css-foyide.*')   # define a regex that matches the class name of the HTML elements that contain the reviews
Review_count = soup.find_all("p", {"class": regex_count}) # find all HTML elements that match the regex
Review_count = Review_count[0].text
Review_count = int(Review_count.split()[0]) # get the total number of reviews

# scraping all reviews of a specific restaurant from yelp:
all_reviews = []
for i in range(0, math.floor(Review_count / 10) - 1):
    url = url_home +"?start="+str(10*i)
    r = requests.get(url, verify=False) # get the HTML of the page
    soup = BeautifulSoup(r.text, "html.parser") # parse the HTML
    regex = re.compile('.*comment.*')   # define a regex that matches the class name of the HTML elements that contain the reviews
    results = soup.find_all("p", {"class": regex}) # find all HTML elements that match the regex
    reviews = [result.text for result in results] # extract the text from the HTML elements
    all_reviews.append(reviews)


# 5. Load Reviews into DataFrame and Score
# ---------------------------------------
all_reviews = sum(all_reviews, []) # flatten the list of lists

df = pd.DataFrame(np.array(all_reviews).flatten(), columns=["review"]) # load the reviews into a dataframe, with the column name "review"

def sentiment_score(all_reviews):
    """Calculate the sentiment score of a review"""
    tokens = tokenizer.encode(all_reviews, return_tensors="pt")
    result = model(tokens)
    sentiment = int(torch.argmax(result.logits)) + 1
    return sentiment

# sentiment_score(df['review'].iloc[0]) # test the function; returns the sentiment score of the first review

df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512])) # apply the function to all reviews and 
# add the sentiment score to the dataframe; Note that the model can only handle 512 tokens at a time, 
# so we slice the review to the first 512 characters


# 6. Plot Results:
# ----------------

score_data = df['sentiment'].to_numpy() # convert the sentiment scores to a numpy array

fig = plt.figure(figsize=(7,7)) # create a figure
ax1 = fig.add_subplot() # add a subplot to the figure
ax1_bars = [1,2,3,4,5] # define the x-axis 

 # plot the histogram:
ax1.hist( 
    score_data, 
    bins=[x for i in ax1_bars for x in (i-0.4,i+0.4)], 
    color='#404080'
    )

ax1.set_xticks(ax1_bars) # set the x-axis ticks
ax1.set_xticklabels(['1','2','3','4','5']) # set the x-axis labels
ax1.set_title("Sentiment Analysis") # set the title
# ax1.set_yscale('log')
ax1.set_ylabel('Reviews') # set the y-axis label

fig.tight_layout() # make the plot look nice
# fig.update_layout(font_size=20)
plt.show() # show the plot





1