import clip
import easyocr
import torch
from PIL import Image
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def clean_tweet(tweet):
    spaced_letter_pattern = r'\b(?:\w\s){2,}\w\b'

    def join_letters(match):
        return match.group().replace(' ', '')
    
    if not isinstance(tweet, str):
        tweet = ""

    else:

        tweet = re.sub(spaced_letter_pattern, join_letters, tweet)
        tweet = tweet.replace('RT', '') # Remove 'RT' from the data
        tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
        tweet = re.sub(r'@\w+', '', tweet)     # Remove mentions
        tweet = re.sub(r'#\w+', '', tweet)     # Remove hashtags
        tweet = re.sub(r'[^A-Za-z\s]', '', tweet)  # Remove special characters
        tweet = re.sub(r'[\w.-]+\.com', '', tweet)
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        tweet = ' '.join(tweet.split()) # For handling extra spaces
    
    return tweet.strip()

# cleaned_tweets = [clean_tweet(tweet) for tweet in raw_data['text_corrected'].astype(str)]
# cleaned_tweets = [tweet.lower() for tweet in cleaned_tweets]


def image_preprocess(image_path, text_available = False, text = None):
    model, preprocess = clip.load("ViT-B/32", device=device)

    if text_available == False:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_path, detail=0)
        if result:
            result = ' '.join(result)
            extracted_text = clean_tweet(result)
        else:
            text_features = torch.zeros((1, model.text_projection.shape[1]), device=device)
    else:
        result = text
        extracted_text = clean_tweet(result)
    
    text_tokens = clip.tokenize([extracted_text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
      image_features = model.encode_image(image)

    audio_features = torch.zeros((1, 512))
    return text_features, audio_features, image_features
