import openai
import re
import numpy as np
from tqdm import tqdm

from llama_index.evaluation import CorrectnessEvaluator
from llama_index.llms import OpenAI
from llama_index import ServiceContext


def evaluate_retrieval(
    llama_index_retriever, 
    queries, 
    golden_sources
):
    results = []

    for query, expected_source in tqdm(list(zip(queries, golden_sources))):
        retrieved_nodes = llama_index_retriever.retrieve(query)
        retrieved_sources = [node.metadata['source'] for node in retrieved_nodes]
        
        # If our label does not include a section, then any sections on the page should be considered a hit.
        if "#" not in expected_source:
            retrieved_sources = [source.split("#")[0] for source in retrieved_sources]
        
        if expected_source in retrieved_sources:
            is_hit = True
            score = retrieved_nodes[retrieved_sources.index(expected_source)].score
        else:
            is_hit = False
            score = 0.0
        
        result = {
            "is_hit": is_hit,
            "score": score,
            "retrieved": retrieved_sources,
            "expected": expected_source,
            "query": query,
        }
        results.append(result)
    return results


def get_hit_rate(results):
    return np.mean([r["is_hit"] for r in results])


def evaluate_e2e(
    llama_index_query_engine, 
    queries, 
    golden_responses, 
    llm=None,
    verbose=False,
):
    # run inference
    if verbose:
        print('Running inference')
        
    generated_responses_str = []
    for query in tqdm(queries):
        response = llama_index_query_engine.query(query)
        generated_responses_str.append(response.response)

    # setup evaluator
    eval_llm = llm or OpenAI(model='gpt-4', temperature=0.0)
    service_context = ServiceContext.from_defaults(llm=eval_llm)
    evaluator = CorrectnessEvaluator(service_context=service_context)

    # run evaluation
    if verbose:
        print('Running eval')
        
    eval_results = []
    for query, rag_response, golden_response in tqdm(list(zip(queries, generated_responses_str, golden_responses))):
        eval_result = evaluator.evaluate(
            query=query, 
            reference=golden_response, 
            response=rag_response)
        eval_results.append(eval_result)
    
    return eval_results


def get_mean_score(results):
    return np.mean([r.score for r in results])