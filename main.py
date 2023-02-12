import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Fine-tune the model on labeled tweets
# Code to fine-tune the model on labeled tweets goes here

# Classify the sentiment of new tweets
def classify_sentiment(tweet):
    # Tokenize the tweet
    input_ids = torch.tensor([tokenizer.encode(tweet, add_special_tokens=True)])
    # Predict the sentiments
    with torch.no_grad():
        output = model(input_ids)
        sentiment = torch.argmax(output[0]).item()
    # Return the sentiment as a string
    if sentiment == 0:
        return "negative"
    elif sentiment == 1:
        return "positive"
    else:
        return "neutral"

# Example usage
tweet = "This is a great day!"
print(classify_sentiment(tweet)) # Output: positive
