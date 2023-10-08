import json
import random
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import Anyscale, OpenAI

from data import get_embedding_model, get_postgres_store, EMBEDDING_DIMENSIONS

    
def _get_vector_store_index(
    service_context,
    embedding_model_name,
):

    embed_dim = EMBEDDING_DIMENSIONS[embedding_model_name]
    vector_store = get_postgres_store(embed_dim)
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        service_context=service_context
    )
    return index


def get_query_engine(
    llm_model_name: str = "meta-llama/Llama-2-70b-chat-hf",
    temperature: float = 0.1,
    embedding_model_name = "text-embedding-ada-002",
    similarity_top_k=2
):
    embed_model = get_embedding_model(embedding_model_name)

    if "llama" in llm_model_name:
        llm = Anyscale(model=llm_model_name, temperature=temperature)
    else:
        llm = OpenAI(model=llm_model_name, temperature=temperature)
    
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    index = _get_vector_store_index(service_context, embedding_model_name)
    return index.as_query_engine(similarity_top_k=similarity_top_k)


def get_retriever(    
    embedding_model_name = "text-embedding-ada-002",
    similarity_top_k=2
):

    embed_model = get_embedding_model(embedding_model_name)
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)

    index = _get_vector_store_index(service_context, embedding_model_name)
    return index.as_query_engine(similarity_top_k=similarity_top_k)


def train_test_split(data, split_ratio=0.8):
    """
    Split a list of items into training and testing sets.

    Args:
        data (list): The list of items to be split.
        split_ratio (float): The ratio of items to include in the training set (default is 0.8).

    Returns:
        tuple: A tuple containing two lists - the training set and the testing set.
    """
    if not 0 <= split_ratio <= 1:
        raise ValueError("Split ratio must be between 0 and 1")

    # Shuffle the data to ensure randomness in the split
    random.shuffle(data)

    # Calculate the split indices
    split_index = int(len(data) * split_ratio)

    # Split the data into training and testing sets
    train_set = data[:split_index]
    test_set = data[split_index:]

    return train_set, test_set


def subsample(data, ratio):
    """
    Subsample a list to a given ratio.

    Args:
        data (list): The list of items to be subsampled.
        ratio (float): The ratio of items to retain in the subsample.

    Returns:
        list: A subsampled list containing the specified ratio of items.
    """
    if not 0 <= ratio <= 1:
        raise ValueError("Ratio must be between 0 and 1")

    # Calculate the number of items to retain in the subsample
    num_items_to_retain = int(len(data) * ratio)

    # Randomly select items to retain
    subsampled_data = random.sample(data, num_items_to_retain)

    return subsampled_data


def write_jsonl(filename, data):
    """
    Write a list of dictionaries to a JSON Lines (JSONL) file.

    Args:
        filename (str): The name of the JSONL file to write to.
        data (list): A list of dictionaries to write as JSONL objects.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')