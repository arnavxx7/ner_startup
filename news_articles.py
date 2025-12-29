
from newsapi import NewsApiClient
from snowflake.snowpark.session import Session
import json
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
pd.set_option("display.max_columns", 20)
# connection_params = {
#     "account": os.getenv("SF_ACCOUNT"),
#     "user": os.getenv("SF_USER"),
#     "password": os.getenv("SF_PASSWORD"),
#     "role": os.getenv("SF_ROLE"),
#     "warehouse": os.getenv("SF_WAREHOUSE"),
#     "database": os.getenv("SF_DATABASE"),
#     "schema": os.getenv("SF_SCHEMA")
# }

# #Starting snowflake session
# session = Session.builder.configs(connection_params).create()
# #Creating snowpark dataframe from pandas dataframe
# sp_df = session.create_dataframe(df)

# # Checking if table already exists

# table_exists = session.sql("SHOW TABLES LIKE 'NEWS_ARTICLES'").collect()

# #if table does not exist
# if len(table_exists)==0:
#     print("Table does not exist - creating table")
#     sp_df.write.mode("overwrite").save_as_table("NEWS_ARTICLES")
# #if table does exist
# else:
#     print("Table does exist - appending data to table")
#     sp_df.write.mode("append").save_as_table("NEWS_ARTICLES")

# session.close()

import worldnewsapi
from worldnewsapi.rest import ApiException


configuration = worldnewsapi.Configuration(
    host = "https://api.worldnewsapi.com"
)
configuration.api_key['apiKey'] = os.getenv("NEWS_API_KEY")
configuration.api_key['headerApiKey'] = os.getenv("NEWS_API_KEY")
with worldnewsapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = worldnewsapi.NewsApi(api_client)
    text = "Startups OR Startup Founders OR Startup funding OR Startup revenue OR VC funding startup"
    language = "en"
    try:
        # Search News
        api_response = api_instance.search_news(text=text, language=language)

        print("The response of NewsApi->search_news:\n")
        news_articles_dict= api_response.to_dict()
        print(news_articles_dict.keys())

    except Exception as e:
        print("Exception when calling NewsApi->search_news: %s\n" % e)

df = pd.json_normalize(news_articles_dict["news"])

print(df.columns)
df["content"] = df.apply(lambda x: x["title"]+"\n"+x["text"], axis=1)

print(df.content)
df.to_csv("artifacts/news_articles.csv", index=False, header=True)

