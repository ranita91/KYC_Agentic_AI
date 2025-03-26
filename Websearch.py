# Google/ Bing:
import pandas as pd
from langchain_community.utilities import SerpAPIWrapper
search = SerpAPIWrapper(serpapi_api_key= "d7cebdfabd8c95799d3ef2f6a154c6101350fe1e26f7b98f00e15eb46b193301") # type: ignore
x=search.run("Fetch me the negative news about Vijay mallaya")
# print(search.run("Fetch me the negative news about Modi"))

# 
web_content=[]
web_content.append(x)
print(web_content)

# https://www.freecodecamp.org/news/web-scraping-python-tutorial-how-to-scrape-data-from-a-website/