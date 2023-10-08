# Building, Evaluating, and Optimizing your RAG App for Production

Large Language Models (LLMs) are revolutionizing how users can search for, interact with, and generate new content. Some recent stacks and toolkits around Retrieval-Augmented Generation (RAG) have emerged, enabling users to build applications such as chatbots using LLMs on their private data. However, while setting up a naive RAG stack is straightforward, having it meet a production quality bar is hard. To be an AI engineer, you need to learn principled development practices for evaluation and optimization of your RAG app - from data parameters to retrieval algorithms to fine-tuning.

This workshop will guide you through this development process. You'll start with the basic RAG stack, create an initial evaluation suite, and then experiment with different advanced techniques to improve RAG performance.

## Environment Setup
Setup python environment
1. Create and activate a python virtual environment
```
python3 -m venv rag
source rag/bin/activate
```
2. Install dependencies
```
pip install -r requirements.txt 
```

Setup postgres
1. Install docker: follow OS-specific instructions at https://docs.docker.com/engine/install/
2. Launch postgres with docker compose (under project directory)
```
docker-compose up -d
```

Prepare OpenAI credentials 
1. Create one at https://platform.openai.com/account/api-keys if you don't have one

## Get Started
We will be going through 3 notebooks, to follow along:
```
jupyter lab
```


## Core Dependencies
```
llama-index
ray[data]

# for notebooks
jupyter

# for postgres
sqlalchemy[asyncio]
pgvector
psycopg2-binary
asyncpg
```
