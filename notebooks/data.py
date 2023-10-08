import numpy as np

import os
import ray
from ray.data import ActorPoolStrategy
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index.readers import HTMLTagReader
from llama_index.vector_stores import PGVectorStore
from llama_index.node_parser import SimpleNodeParser

EMBEDDING_DIMENSIONS = {
    'thenlper/gte-base': 768,
    'BAAI/bge-large-en': 1024,
    'text-embedding-ada-002': 1536
}

def path_to_uri(path, scheme="https://", domain="docs.ray.io"):
    # Converts the file path of a Ray documentation page to the original URL for the documentation.
    # Example: /efs/shared_storage/goku/docs.ray.io/en/master/rllib-env.html -> https://docs.ray.io/en/master/rllib/rllib-env.html#environments
    return scheme + domain + str(path).split(domain)[-1]


def extract_sections(record):
    # Given a HTML file path, extract out text from the section tags, and return a LlamaIndex document from each one. 
    html_file_path = record["path"]
    reader = HTMLTagReader(tag="section")
    documents = reader.load_data(html_file_path)
    
    # For each document, store the source URL as part of the metadata.
    for document in documents:
        document.metadata["source"] = f"{path_to_uri(document.metadata['file_path'])}#{document.metadata['tag_id']}"
    return [{"document": document} for document in documents]


def get_embedding_model(model_name, embed_batch_size=100):
    if model_name == "text-embedding-ada-002":
            return OpenAIEmbedding(
                model=model_name,
                embed_batch_size=embed_batch_size,
                api_key=os.environ["OPENAI_API_KEY"])
    else:
        return HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=embed_batch_size
        )
    
    
class EmbedChunks:
    def __init__(self, model_name):
        self.embedding_model = get_embedding_model(model_name)
    
    def __call__(self, node_batch):
        # Get the batch of text that we want to embed.
        nodes = node_batch["node"]
        text = [node.text for node in nodes]
        
        # Embed the batch of text.
        embeddings = self.embedding_model.get_text_embedding_batch(text)
        assert len(nodes) == len(embeddings)

        # Store the embedding in the LlamaIndex node.
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return {"embedded_nodes": nodes}


def get_postgres_store(embed_dim=768):
    return PGVectorStore.from_params(
            database="postgres", 
            user="postgres", 
            password="postgres", 
            host="localhost", 
            table_name="document",
            port="5432",
            embed_dim=embed_dim,
        )


class StoreResults:
    def __init__(self, embed_dim=768):
        self.vector_store = get_postgres_store(embed_dim)
    
    def __call__(self, batch):
        embedded_nodes = batch["embedded_nodes"]
        self.vector_store.add(list(embedded_nodes))
        return {}

def create_nodes(docs_path, chunk_size, chunk_overlap):
    ds = ray.data.from_items(
        [{"path": path} for path in docs_path.rglob("*.html") if not path.is_dir()]
    )
    sections_ds = ds.flat_map(extract_sections)

    def chunk_document(document):
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents([document["document"]])
        return [{"node": node} for node in nodes]

    chunks_ds = sections_ds.flat_map(chunk_document, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False))

    return chunks_ds
    
def build_index(docs_path, embedding_model_name, chunk_size, chunk_overlap):

    chunks_ds = create_nodes(docs_path, chunk_size, chunk_overlap)

    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model_name},
        batch_size=100, 
        num_gpus=0 if embedding_model_name!="text-embedding-ada-002" else 0,
        compute=ActorPoolStrategy(size=2))

    # Index data
    embed_dim=EMBEDDING_DIMENSIONS[embedding_model_name]
    embedded_chunks.map_batches(
        StoreResults,
        fn_constructor_kwargs={"embed_dim": embed_dim},
        batch_size=128,
        num_cpus=1,
        compute=ActorPoolStrategy(size=8),
        # Since our database is only created on the head node, we need to force the Ray tasks to only executed on the head node.
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)

    ).count()