#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity  # measure similarity between two vectors, to assess how alike two data points are
from sklearn.feature_extraction.text import TfidfVectorizer #TfidfVectorizer converts text data into a matrix of numerical features

import os #Importing the os module makes these functions available in your code.
from scipy.sparse import coo_matrix


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required resources if missing
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Punkt tokenizer
try:
    word_tokenize("test sentence")
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Add this line



# In[2]:


data = pd.read_csv("marketing_sample_data.tsv", sep = '\t')  #seperated


# In[3]:


data = data[['Uniq Id','Crawl Timestamp','Product Id', 'Product Rating', 'Product Reviews Count', 'Product Category', 'Product Brand', 'Product Name', 'Product Image Url', 'Product Url', 'Product Description','Product Available Inventory', 'Product Tags']]


data["Product Rating"] = data["Product Rating"].fillna(0)
data["Product Reviews Count"] = data["Product Reviews Count"].fillna(0)
data["Product Category"] = data["Product Category"].fillna('')
data["Product Brand"] = data["Product Brand"].fillna('')
data["Product Description"] = data["Product Description"].fillna('')
data["Product Url"] = data["Product Url"].fillna('')


# In[4]:


data.duplicated().sum() #no duplicate value in data


# In[5]:


column_name = {
    'Uniq Id': 'ID',
    'Crawl Timestamp' :'Time',
    'Product Id': 'ProdID',
    'Product Rating': 'Rating',
    'Product Reviews Count': 'ReviewCount',
    'Product Category': 'Category',
    'Product Brand': 'Brand',
    'Product Name': 'Name',
    'Product Image Url': 'ImageURL',
    'Product Url': 'ProductURL',
    'Product Description': 'Description',
    'Product Available Inventory' : 'ProductInventory',
    'Product Tags': 'Tags',
    'Product Contents': 'Contents'
}

data.rename(columns=column_name,inplace=True)


# In[6]:


data['ID'] = data['ID'].str.extract(r'(\d+)').astype(float)
data['ProdID'] = data['ProdID'].str.extract(r'(\d+)').astype(float)


# In[7]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK data if not already
#nltk.download('punkt')
#nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_and_extract_tags_nltk(text):
    # Lowercase, tokenize
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords, keep alphanumeric tokens only
    tags = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ', '.join(tags)

columns_to_extract_tags_from = ['Category', 'Brand', 'Description']

for column in columns_to_extract_tags_from:
    data[column] = data[column].fillna('')
    data[column + '_tags'] = data[column].apply(clean_and_extract_tags_nltk)


# In[8]:


data['Tags'] = data[columns_to_extract_tags_from].apply(lambda row: ', '.join(row), axis=1)


# In[9]:


average_ratings =data.groupby(['Name', 'ReviewCount', 'Brand', 'ImageURL'])['Rating'].mean().reset_index()
top_rated_items = average_ratings.sort_values(by='Rating', ascending=False)

rating_base_recommendation = top_rated_items.head(10)

rating_base_recommendation.loc[:, 'Rating'] = rating_base_recommendation['Rating'].astype(int)
rating_base_recommendation.loc[:, 'ReviewCount'] = rating_base_recommendation['ReviewCount'].astype(int)

print("Rating Base Recommendation System: (Trending Products)")
rating_base_recommendation[['Name', 'Rating', 'ReviewCount', 'Brand', 'ImageURL']]


# In[10]:


from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import Image, display
import pandas as pd

# 1. Create user-item matrix
user_item_matrix = data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0).astype(int)

# 2. Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# 3. Main collaborative filtering recommendation function with optional filters
def collaborative_recommendations(data, target_user_id, top_n=10, filter_stock=True, filter_recent=True, recent_days=30):
    if target_user_id not in user_item_matrix.index:
        print(f"User {target_user_id} not found. Showing top rated products overall:")
        return get_top_rated_items(data, top_n)
    
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]
    
    # Exclude the user themself
    similar_user_indices = user_similarities.argsort()[::-1][1:]
    
    recommend_items = []
    
    for user_index in similar_user_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (user_item_matrix.iloc[target_user_index] == 0) & (rated_by_similar_user > 0)
        
        new_items = user_item_matrix.columns[not_rated_by_target_user]
        recommend_items.extend(new_items)
        
        if len(recommend_items) >= top_n:
            break

    # Drop duplicates and select top_n
    recommend_items = list(dict.fromkeys(recommend_items))[:top_n]

    if not recommend_items:
        print(f"No collaborative recommendations for user {target_user_id}. Showing top rated products:")
        return get_top_rated_items(data, top_n)
    
    # Filter recommended items from original data
    recs = data[data['ProdID'].isin(recommend_items)].drop_duplicates(subset='Name')
    
    # Optional filters
    if filter_stock and 'Product Available Inventory' in recs.columns:
        recs = recs[recs['Product Available Inventory'] > 0]
    
    if filter_recent and 'CrawlTimestamp' in recs.columns:
        import datetime
        cutoff_date = pd.Timestamp.today() - pd.Timedelta(days=recent_days)
        recs = recs[pd.to_datetime(recs['CrawlTimestamp']) >= cutoff_date]
    
    # Sort by Rating descending and get top_n
    recs = recs.sort_values(by='Rating', ascending=False).head(top_n)
    
    return recs[['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

# 4. Fallback: top-rated items overall
def get_top_rated_items(data, top_n=10):
    return data.sort_values(by='Rating', ascending=False)[
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
    ].drop_duplicates(subset='Name').head(top_n)

# 5. Display recommendations nicely with images in Jupyter Notebook
def display_recommendations_with_images(recommendations_df):
    for idx, row in recommendations_df.iterrows():
        print(f"Name: {row['Name']}")
        print(f"Brand: {row['Brand']}")
        print(f"Rating: {row['Rating']}")
        print(f"Review Count: {row['ReviewCount']}")
        # To actually display images in Jupyter, uncomment the next line
        display(Image(url=row['ImageURL'], width=120))
        print("-" * 50)

# Example usage


# In[11]:


import gradio as gr
import pandas as pd
import html
import uuid
from sklearn.metrics.pairwise import cosine_similarity

# --- STEP 1: Load your data (you already did this before) ---
# data = pd.read_csv("your_data.csv") ← you already loaded this
data = data[data['ID'] >= 0]

user_item_matrix = data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0).astype(int)
user_similarity = cosine_similarity(user_item_matrix)
user_id_list = sorted(user_item_matrix.index.tolist())

# --- STEP 2: Create user-item matrix and compute user similarity ---


# --- STEP 3: Collaborative Filtering Function ---
def collaborative_recommendations(data, target_user_id, top_n=7):
    try:
        if target_user_id not in user_item_matrix.index:
            return get_top_rated_items(data, top_n)

        target_user_index = user_item_matrix.index.get_loc(target_user_id)
        user_similarities = user_similarity[target_user_index]
        similar_user_indices = user_similarities.argsort()[::-1][1:]

        recommend_items = []
        for user_index in similar_user_indices:
            rated_by_similar_user = user_item_matrix.iloc[user_index]
            not_rated_by_target_user = (user_item_matrix.iloc[target_user_index] == 0) & (rated_by_similar_user > 0)
            new_items = user_item_matrix.columns[not_rated_by_target_user]
            recommend_items.extend(new_items)
            if len(recommend_items) >= top_n:
                break

        recommend_items = list(dict.fromkeys(recommend_items))[:top_n]

        if not recommend_items:
            return get_top_rated_items(data, top_n)

        return data[data['ProdID'].isin(recommend_items)][
            ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'ProductURL', 'Rating']
        ].drop_duplicates(subset='Name').sort_values(by='Rating', ascending=False).head(top_n)

    except Exception as e:
        return pd.DataFrame([{"Name": f"Error: {str(e)}"}])

# --- STEP 4: Fallback to top-rated items ---
def get_top_rated_items(data, top_n=7):
    return data.sort_values(by='Rating', ascending=False)[
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'ProductURL', 'Rating']
    ].drop_duplicates(subset='Name').head(top_n)

# --- STEP 5: Image Slider Builder ---
def build_image_slider(image_urls, slider_id):
    total = len(image_urls)
    if total == 1:
        return f'''
            <div style="width:240px; height:240px; margin:auto;">
                <img src="{image_urls[0]}" style="width:100%; height:100%; object-fit:contain; border-radius:8px; border:1px solid #ccc;" />
            </div>
        '''
    radios = ''.join([
        f'<input type="radio" name="slider-{slider_id}" id="slide-{slider_id}-{i}" {"checked" if i == 0 else ""} style="display:none;">'
        for i in range(total)
    ])
    slides_html = ''
    for i, url in enumerate(image_urls):
        prev = f"slide-{slider_id}-{(i - 1) % total}"
        nxt = f"slide-{slider_id}-{(i + 1) % total}"
        slides_html += f'''
        <div class="slide-{slider_id}" id="content-{slider_id}-{i}">
            <label for="{prev}" class="arrow arrow-left">❮</label>
            <img src="{url}" />
            <label for="{nxt}" class="arrow arrow-right">❯</label>
        </div>
        '''
    css_rules = ''.join([
        f'#slide-{slider_id}-{i}:checked ~ .slider-{slider_id} #content-{slider_id}-{i} {{ display: flex; }}'
        for i in range(total)
    ])
    return '''
        <div style="margin:auto; width:240px;">
            {radios}
            <div class="slider-{id}" style="position:relative; width:240px; height:240px; margin:auto;">
                {slides}
            </div>
            <style>
                .slider-{id} div {{
                    display: none;
                    justify-content: space-between;
                    align-items: center;
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 240px;
                    height: 240px;
                }}
                .slider-{id} img {{
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                    border-radius: 8px;
                    border: 1px solid #ccc;
                }}
                .arrow {{
                    font-size: 24px;
                    color: black;
                    background: rgba(255,255,255,0.7);
                    padding: 4px 8px;
                    cursor: pointer;
                    user-select: none;
                    z-index: 10;
                }}
                .arrow-left {{
                    position: absolute;
                    left: -12px;
                    top: 40%;
                }}
                .arrow-right {{
                    position: absolute;
                    right: -12px;
                    top: 40%;
                }}
                {rules}
            </style>
        </div>
    '''.format(radios=radios, slides=slides_html, id=slider_id, rules=css_rules)

# --- STEP 6: Build Gradio Output HTML ---
def collaborative_ui(user_id):
    try:
        rec_df = collaborative_recommendations(data, user_id, top_n=7)
        if rec_df.empty:
            return "<b>No recommendations found.</b>"

        output_html = []
        for _, row in rec_df.iterrows():
            raw_links = str(row.get('ImageURL', '')).strip()
            if raw_links.startswith('(') and raw_links.endswith(')'):
                raw_links = raw_links[1:-1]
            links = [link.strip() for link in raw_links.split('|')]
            valid_links = [link for link in links if link.startswith('http')]
            if not valid_links:
                valid_links = ["https://via.placeholder.com/240?text=No+Image"]

            slider_id = str(uuid.uuid4())[:8]
            image_slider = build_image_slider(valid_links, slider_id)

            name = html.escape(str(row.get('Name', '')))
            brand = html.escape(str(row.get('Brand', '')))
            rating = html.escape(str(row.get('Rating', '')))
            reviews = html.escape(str(row.get('ReviewCount', '')))
            product_url = str(row.get("ProductURL", "")).strip()
            link_html = (
                f'<a href="{product_url}" target="_blank" rel="noopener noreferrer" style="color:blue;">🔗 View Product</a>'
                if product_url.startswith("http") else
                "<div style='color:gray;'>🔗 No Product URL</div>"
            )

            block = f"""
            <div style='margin-bottom:50px; text-align:center;'>
                {image_slider}<br>
                <strong>{name}</strong><br>
                Brand: {brand}<br>
                Rating: {rating}<br>
                Reviews: {reviews}<br>
                {link_html}
            </div>
            """
            output_html.append(block)

        return "<hr>".join(output_html)

    except Exception as e:
        return f"<b>Error:</b> {str(e)}"

# --- STEP 7: Launch Gradio Interface ---
user_id_list = sorted(user_item_matrix.index.tolist())




# In[ ]:




