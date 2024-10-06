import os
import wget
import zipfile
import chromadb
import pandas as pd
from ast import literal_eval
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from constants import EMBEDDING_MODEL


def download():
    # Download the zip and extract it.
    # embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"
    # wget.download(embeddings_url)
    # with zipfile.ZipFile(
    #     "vector_database_wikipedia_articles_embedded.zip", "r"
    # ) as zip_ref:
    #     zip_ref.extractall("../data")
    pass


def query_collection(collection, query, max_results, dataframe):
    print("query_collection")
    results = collection.query(
        query_texts=query, n_results=max_results, include=["distances"]
    )
    df = pd.DataFrame(
        {
            "id": results["ids"][0],
            "score": results["distances"][0],
            "title": dataframe[dataframe.vector_id.isin(results["ids"][0])]["title"],
            "content": dataframe[dataframe.vector_id.isin(results["ids"][0])]["text"],
        }
    )

    return df


def create_wikipedia_collection():
    article_df = pd.read_csv("../data/vector_database_wikipedia_articles_embedded.csv")
    print(article_df.head())
    article_df["title_vector"] = article_df.title_vector.apply(literal_eval)
    article_df["content_vector"] = article_df.content_vector.apply(literal_eval)
    article_df["vector_id"] = article_df["vector_id"].apply(str)
    article_df = article_df[:10]

    chroma_client = chromadb.EphemeralClient()
    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL
    )
    wikipedia_content_collection = chroma_client.create_collection(
        name="wikipedia_content", embedding_function=embedding_function
    )
    wikipedia_title_collection = chroma_client.create_collection(
        name="wikipedia_titles", embedding_function=embedding_function
    )

    print("Collections created")

    wikipedia_content_collection.add(
        ids=article_df.vector_id.tolist(),
        embeddings=article_df.content_vector.tolist(),
    )

    wikipedia_title_collection.add(
        ids=article_df.vector_id.tolist(),
        embeddings=article_df.title_vector.tolist(),
    )

    print("Vectors added")

    title_query_result = query_collection(
        collection=wikipedia_title_collection,
        query="modern art in Europe",
        max_results=10,
        dataframe=article_df,
    )
    print("title_query_result.head()", title_query_result.head())
    content_query_result = query_collection(
        collection=wikipedia_content_collection,
        query="Famous battles in Scottish history",
        max_results=10,
        dataframe=article_df,
    )
    print("content_query_result.head()", content_query_result.head())


create_wikipedia_collection()
