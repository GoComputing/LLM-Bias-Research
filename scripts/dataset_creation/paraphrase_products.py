from tools import load_llm, paraphrase_text
from recommender import base_prompt_template

from searchengine import AmazonSearchEngine
from dotenv import load_dotenv
from tqdm.auto import tqdm
from copy import deepcopy
import argparse
import random
import json
import os

models = {
    # https://cloud.google.com/vertex-ai/generative-ai/docs/models
    'google': [
        'google:gemini-2.0-flash-lite',
        'google:gemini-2.0-flash',
    ],
    # https://docs.anthropic.com/en/docs/about-claude/models/overview
    'anthropic': [
        'anthropic:claude-3-7-sonnet-20250219',
        'anthropic:claude-3-5-haiku-20241022'
    ],
    # https://python.langchain.com/api_reference/_modules/langchain_community/callbacks/openai_info.html
    'openai': [
        'openai:o4-mini-2025-04-16',
        'openai:o3-mini-2025-01-31'
    ],
    # https://ollama.com/library/llama3.1/tags
    'llama3.1': [
        'llama3.1:8b-instruct-fp16',
        'llama3.1:70b-instruct-q8_0'
    ],
    # https://ollama.com/library/deepseek-r1/tags
    'deepseek': [
        'deepseek-r1:8b-llama-distill-fp16',
        'deepseek-r1:7b-qwen-distill-fp16',
    ]
}

def main(args):

    load_dotenv()

    input_queries_path = args.input_queries_path
    search_engine_path = args.search_engine_path
    output_path = args.output_path
    seed = args.seed
    samples_per_query = args.samples_per_query

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load input dataset
    with open(input_queries_path, 'r') as f:
        input_queries = json.load(f)

    # Retrieve articles
    search_results_path = os.path.join(output_path, 'search_results.json')

    if not os.path.exists(search_results_path):

        # Load engine
        search_engine = AmazonSearchEngine()
        search_engine.load_search_engine(search_engine_path)
        search_results = []

        # Search products according to given queries
        for query in tqdm(input_queries, desc='Retrieve'):
            top_product = search_engine.query(query, top_k=1).to_dict('records')[0]
            top_product = {'id': top_product['PRODUCT_ID'], 'title': top_product['TITLE'], 'description': top_product['DESCRIPTION']}
            search_results.append({'query': query, 'product': top_product})

        # Store results
        with open(search_results_path, 'w') as f:
            json.dump(search_results, f, indent=3)
    else:
        with open(search_results_path, 'r') as f:
            search_results = json.load(f)

    # Paraphrase articles
    model_results_dir = os.path.join(output_path, 'models_paraphrasing')
    if not os.path.exists(model_results_dir):
        os.mkdir(model_results_dir)
    num_models = sum([len(models_names) for models_names in models.values()])
    models_progress_bar = tqdm(total=num_models, desc='Models', position=0)

    for model_group, models_names in models.items():
        for model_name in models_names:

            model_result_path = os.path.join(model_results_dir, model_name+'.json')
            if os.path.exists(model_result_path):
                models_progress_bar.update(1)
                continue

            paraphraser_llm = load_llm(model_name)
            paraphrase_results = {
                'metadata': {
                    'model_group': model_group,
                    'model_name': model_name
                },
                'results': []
            }

            for query_info in tqdm(search_results, desc='Paraphrase', position=1):
                new_description, paraphrase_data = paraphrase_text(
                    paraphraser_llm,
                    query_info['product']['description'],
                    return_raw_response=True, original_on_failure=True)

                query_info = deepcopy(query_info)
                query_info['product']['description'] = new_description
                query_info['paraphrased_data'] = paraphrase_data
                paraphrase_results['results'].append(query_info)

            # Store model paraphrasing results
            with open(model_result_path, 'w') as f:
                json.dump(paraphrase_results, f, indent=3)

            models_progress_bar.update(1)

    # Load paraphrased products
    paraphrased_products = []
    for model_group, models_names in models.items():
        for model_name in models_names:
            paraphrased_path = os.path.join(model_results_dir, model_name+'.json')
            with open(paraphrased_path, 'r') as f:
                model_results = json.load(f)
            for query_info in model_results['results']:
                query_info['product']['model_group'] = model_group
                query_info['product']['model_name'] = model_name
            paraphrased_products.append(model_results['results'])

    # Generate final dataset
    dataset = {
        'prompt_template': base_prompt_template(),
        'search_engine_path': search_engine_path,
        'queries_path': input_queries_path,
        'models': models,
        'seed': seed,
        'samples_per_query': samples_per_query,
        'queries': []
    }

    random.seed(seed)

    for products in zip(*paraphrased_products):
        for i in range(samples_per_query):
            query_info = {
                'query': products[0]['query'],
                'products': [product['product'] for product in products]
            }
            random.shuffle(query_info['products'])
            dataset['queries'].append(query_info)

    # Store dataset
    dataset_path = os.path.join(output_path, 'dataset.json')
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate a dataset using several LLMs as paraphrasers')
    parser.add_argument('-i', '--input-queries-path', required=True, type=str, help='Path to a JSON-like array of queries')
    parser.add_argument('-s', '--search-engine-path', required=True, type=str, help='Path to the directory holding the search engine (generated by script create_index.py)')
    parser.add_argument('-o', '--output-path', required=True, type=str, help='Path to a directory where results will be stored')
    parser.add_argument('--seed', type=int, default=5243534, help='Seed used for deterministic output')
    parser.add_argument('-n', '--samples-per-query', type=int, default=10, help='Number of samples for each query. Each sample will randomly shuffle the products associated to the query')

    args = parser.parse_args()

    main(args)
