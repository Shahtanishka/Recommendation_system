#!/usr/bin/env python
# coding: utf-8

# In[2]:
#!/usr/bin/env python
# coding: utf-8



import gradio as gr
from content_based import recommend_products
from collaborative import collaborative_ui
from collaborative import user_id_list

with gr.Blocks() as demo:
    gr.Markdown("## 🛍️ Product Recommendation System")

    with gr.Tab("Content-Based Recommender"):
        product_input = gr.Textbox(label="Enter a Product Name")
        content_output = gr.HTML(label="Top Recommendations")
        product_input.submit(fn=recommend_products, inputs=product_input, outputs=content_output)

    with gr.Tab("Collaborative Filtering Recommender"):
        user_dropdown = gr.Dropdown(choices=user_id_list, label="Select User ID")
        collaborative_output = gr.HTML(label="Collaborative Recommendations")
        user_dropdown.change(fn=collaborative_ui, inputs=user_dropdown, outputs=collaborative_output)

if __name__ == "__main__":
    demo.launch()

# In[ ]:




