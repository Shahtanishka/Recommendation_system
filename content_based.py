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


data.columns


# In[4]:


data = data[['Uniq Id','Crawl Timestamp','Product Id', 'Product Rating', 'Product Reviews Count', 'Product Category', 'Product Brand', 'Product Name', 'Product Image Url', 'Product Url', 'Product Description','Product Available Inventory', 'Product Tags']]


# In[5]:


data["Product Tags"]


# In[6]:


data.shape


# In[7]:


data["Product Rating"] = data["Product Rating"].fillna(0)
data["Product Reviews Count"] = data["Product Reviews Count"].fillna(0)
data["Product Category"] = data["Product Category"].fillna('')
data["Product Brand"] = data["Product Brand"].fillna('')
data["Product Description"] = data["Product Description"].fillna('')
data["Product Url"] = data["Product Url"].fillna('')



# In[8]:


data.isnull().sum()


# In[9]:


data.duplicated().sum() #no duplicate value in data


# In[10]:


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


# In[11]:


data['ID'] = data['ID'].str.extract(r'(\d+)').astype(float)
data['ProdID'] = data['ProdID'].str.extract(r'(\d+)').astype(float)


# In[12]:


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


# In[13]:


data['Tags'] = data[columns_to_extract_tags_from].apply(lambda row: ', '.join(row), axis=1)


# In[14]:


average_ratings =data.groupby(['Name', 'ReviewCount', 'Brand', 'ImageURL'])['Rating'].mean().reset_index()
top_rated_items = average_ratings.sort_values(by='Rating', ascending=False)

rating_base_recommendation = top_rated_items.head(10)

rating_base_recommendation.loc[:, 'Rating'] = rating_base_recommendation['Rating'].astype(int)
rating_base_recommendation.loc[:, 'ReviewCount'] = rating_base_recommendation['ReviewCount'].astype(int)

print("Rating Base Recommendation System: (Trending Products)")
rating_base_recommendation[['Name', 'Rating', 'ReviewCount', 'Brand', 'ImageURL']]


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import pandas as pd

# 1. Prepare combined text features column (Name + Category + Brand + Description)
data['combined_features'] = (
    data['Name'].fillna('') + ' ' +
    data['Category'].fillna('') + ' ' +
    data['Brand'].fillna('') + ' ' +
    data['Description'].fillna('')
)

# 2. Create TF-IDF matrix and cosine similarity matrix
def create_tfidf_matrix(data, feature_column='combined_features'):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data[feature_column])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf_vectorizer, tfidf_matrix, cosine_sim

tfidf_vectorizer, tfidf_matrix, cosine_sim = create_tfidf_matrix(data)

# 3. Fuzzy match function for close product names
def find_closest_match(name, choices, cutoff=0.6):
    matches = difflib.get_close_matches(name, choices, n=1, cutoff=cutoff)
    return matches[0] if matches else None


# 5. Content-based recommendation function with fallback to keyword search
def content_based_recommendations(data, item_name, cosine_sim, top_n=10):
    if item_name not in data['Name'].values:
        close_match = find_closest_match(item_name, data['Name'].tolist())
        if close_match:
            print(f"Item '{item_name}' not found. Did you mean '{close_match}'? Recommending based on '{close_match}'.")
            item_name = close_match
        else:
            print(f"Showing top rated items matching keyword '{item_name}':")
            return search_top_rated_by_keyword(data, item_name, top_n)
    
    item_index = data[data['Name'] == item_name].index[0]
    similarity_scores = list(enumerate(cosine_sim[item_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar = similarity_scores[1:top_n+1]  # skip the item itself
    indices = [i[0] for i in top_similar]
    
    return data.iloc[indices][['Name', 'ReviewCount', 'Brand', 'ImageURL','ProductURL', 'Rating']]



def search_top_rated_by_keyword(data, keyword, top_n=10):
    # Try exact substring match first (case-insensitive)
    filtered = data[data['combined_features'].str.contains(keyword, case=False, na=False)]
    if filtered.empty:
        # If no exact substring found, try fuzzy match on unique words in combined_features
        all_words = set()
        for text in data['combined_features']:
            all_words.update(text.lower().split())
        close_match = find_closest_match(keyword.lower(), list(all_words))
        if close_match:
            print(f"No exact match found for '{keyword}'. Using closest keyword '{close_match}' for search.")
            filtered = data[data['combined_features'].str.contains(close_match, case=False, na=False)]
        else:
            print(f"No matches found for '{keyword}' or similar keywords.")
            return pd.DataFrame()  # empty result
    filtered_sorted = filtered.sort_values(by='Rating', ascending=False)
    return filtered_sorted[['Name', 'ReviewCount', 'Brand', 'ImageURL','ProductURL', 'Rating']].head(top_n)

# Usage example:
# result = content_based_recommendations(data, "lipstick", cosine_sim, top_n=10)
# print(result)


# In[16]:


import gradio as gr
import html
import uuid

# --- Clean image slider function ---
def build_image_slider(image_urls, slider_id):
    total = len(image_urls)

    if total == 1:
        return f'''
            <div style="width:240px; height:240px; margin:auto;">
                <img src="{image_urls[0]}" style="width:100%; height:100%; object-fit:contain; border-radius:8px; border:1px solid #ccc;" />
            </div>
        '''

    # Radio buttons (hidden)
    radios = ''.join([
        f'<input type="radio" name="slider-{slider_id}" id="slide-{slider_id}-{i}" {"checked" if i == 0 else ""} style="display:none;">'
        for i in range(total)
    ])

    # Slides
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

    # CSS to show selected slide
    css_rules = ''.join([
        f'#slide-{slider_id}-{i}:checked ~ .slider-{slider_id} #content-{slider_id}-{i} {{ display: flex; }}'
        for i in range(total)
    ])

    # Full HTML block
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

# --- Recommender Logic ---
def recommend_products(product_name):
    try:
        result_df = content_based_recommendations(data, product_name, cosine_sim, top_n=7)
        if result_df.empty:
            return "<b>No recommendations found. Try another product name.</b>"

        recommendations = []

        for _, row in result_df.iterrows():
            # Image links
            raw_links = str(row.get('ImageURL', '')).strip()
            if raw_links.startswith('(') and raw_links.endswith(')'):
                raw_links = raw_links[1:-1]

            links = [link.strip() for link in raw_links.split('|')]
            valid_links = [link for link in links if link.startswith('http')]
            if not valid_links:
                valid_links = ["https://via.placeholder.com/240?text=No+Image"]

            slider_id = str(uuid.uuid4())[:8]
            image_slider_html = build_image_slider(valid_links, slider_id)

            # Metadata
            product_name_escaped = html.escape(str(row.get('Name', '')))
            brand = html.escape(str(row.get('Brand', '')))
            rating = html.escape(str(row.get('Rating', '')))
            review_count = html.escape(str(row.get('ReviewCount', '')))
            product_url = str(row.get("ProductURL", "")).strip()

            link_html = (
                f'<a href="{product_url}" target="_blank" rel="noopener noreferrer" '
                f'style="color:blue; text-decoration:underline;">🔗 View Product</a>'
                if product_url.startswith("http") else
                "<div style='color:gray;'>🔗 No Product URL</div>"
            )

            # HTML Block
            info_html = f"""
            <div style='margin-bottom:50px; text-align:center;'>
                {image_slider_html}<br>
                <strong>{product_name_escaped}</strong><br>
                Brand: {brand}<br>
                Rating: {rating}<br>
                Reviews: {review_count}<br>
                {link_html}
            </div>
            """
            recommendations.append(info_html)

        return "<hr>".join(recommendations)

    except Exception as e:
        return f"<b>Error:</b> {str(e)}"

# --- Gradio Interface ---
demo = gr.Interface(
    fn=recommend_products,
    inputs=gr.Textbox(label="Enter a Product Name"),
    outputs=gr.HTML(label="Top Recommendations"),
    title="🛍️ Product Recommendation System",
    description="Enter a product name to get personalized recommendations with product details and preview images."
)


# In[ ]:




