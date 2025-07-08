from tools import load_llm, paraphrase_text
from recommender import base_prompt_template

from searchengine import AmazonSearchEngine
from itertools import permutations
from dotenv import load_dotenv
from tqdm.auto import tqdm
from copy import deepcopy
import argparse
import random
import json
import os

models = {
    # https://cloud.google.com/vertex-ai/generative-ai/docs/models
    # 'google': [
    #     'google:gemini-2.0-flash-lite',
    #     'google:gemini-2.0-flash',
    # ],
    # https://docs.anthropic.com/en/docs/about-claude/models/overview
    # 'anthropic': [
    #     'anthropic:claude-3-7-sonnet-20250219',
    #     'anthropic:claude-3-5-haiku-20241022'
    # ],
    # https://python.langchain.com/api_reference/_modules/langchain_community/callbacks/openai_info.html
    # 'openai': [
    #     'openai:o4-mini-2025-04-16',
    #     'openai:o3-mini-2025-01-31'
    # ],
    # https://ollama.com/library/llama3.1/tags
    # 'llama3.1': [
    #     'llama3.1:8b-instruct-fp16',
    #     'llama3.1:70b-instruct-q8_0'
    # ],
    # https://ollama.com/library/deepseek-r1/tags
    # 'deepseek': [
    #     'deepseek-r1:8b-llama-distill-fp16',
    #     'deepseek-r1:7b-qwen-distill-fp16',
    # ],
    'nomodel': [
        'nomodel:originaltext'
    ]
}

def main(args):

    load_dotenv()

    input_queries_path = args.input_queries_path
    search_engine_path = args.search_engine_path
    output_path = args.output_path
    output_name = args.output_name
    seed = args.seed
    samples_per_query = args.samples_per_query
    top_k = args.top_k
    share_permutations = args.share_permutations

    dataset_path = os.path.join(output_path, f'{output_name}.json')
    if os.path.exists(dataset_path):
        raise IOError(f'Final dataset JSON file already exists ({dataset_path})')

    if top_k is None:
        top_k = 1

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
            top_products = search_engine.query(query, top_k=top_k).to_dict('records')
            for i in range(len(top_products)):
                top_products[i] = {'id': top_products[i]['PRODUCT_ID'], 'title': top_products[i]['TITLE'], 'description': top_products[i]['DESCRIPTION']}
            if top_k == 1:
                top_products = top_products[0]
            search_results.append({'query': query, 'product' if top_k == 1 else 'products': top_products})

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

            paraphraser_llm = load_llm(model_name)
            paraphrase_results = {
                'metadata': {
                    'model_group': model_group,
                    'model_name': model_name,
                },
                'results': []
            }

            model_result_path = os.path.join(model_results_dir, model_name+'.json')
            if os.path.exists(model_result_path):
                with open(model_result_path, 'r') as f:
                    paraphrase_results = json.load(f)

            # The for will recover if the script failed and launched again
            for query_info in tqdm(search_results[len(paraphrase_results['results']):], desc='Paraphrase', position=1):

                query_info = deepcopy(query_info)
                products = [query_info['product']] if 'product' in query_info else query_info['products']
                paraphrased_data = []

                for product in products:
                    new_description, paraphrase_data = paraphrase_text(
                        paraphraser_llm,
                        product['description'],
                        return_raw_response=True, original_on_failure=True)

                    product['description'] = new_description
                    paraphrased_data.append(paraphrase_data)

                if len(paraphrased_data) == 1:
                    paraphrased_data = paraphrased_data[0]

                query_info['paraphrased_data'] = paraphrased_data
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
                products = [query_info['product']] if 'product' in query_info else query_info['products']
                for product in products:
                    product['model_group'] = model_group
                    product['model_name'] = model_name
            paraphrased_products.append(model_results['results'])

    # Generate final dataset
    dataset = {
        'prompt_template': base_prompt_template(),
        'search_engine_path': search_engine_path,
        'queries_path': input_queries_path,
        'models': models,
        'seed': seed,
        'samples_per_query': samples_per_query,
        'top_k': top_k,
        'share_permutations': share_permutations,
        'queries': [],
    }

    random.seed(seed)

    def sample_permutations(permutations_population, quantity):

        if quantity is None:
            permutations_list =  permutations_population
        else:
            indices = [random.randint(0, len(permutations_population)-1) for _ in range(quantity)]
            permutations_list = [permutations_population[indices[i]] for i in range(quantity)]

        return permutations_list

    # Generate all possible permutations
    permutations_population = list(permutations(range(top_k)))

    # Special case: if samples per query is less than 1, use all possible permutations
    permutations_quantity = samples_per_query
    if samples_per_query < 1:
        samples_per_query = len(permutations_population)
        permutations_quantity = None

    # Generate permutations shared across all queries
    query_permutations = None
    if share_permutations:
        query_permutations = sample_permutations(permutations_population, permutations_quantity)

    for products in zip(*paraphrased_products):
        if not share_permutations:
            query_permutations = sample_permutations(permutations_population, permutations_quantity)
        for i in range(samples_per_query):
            query_info = {
                'query': products[0]['query'],
            }
            products = list(products)
            random.shuffle(products) # Shuffle "models" (each item is the paraphrased generated by one model)
            if top_k == 1:
                selected_products = [product['product'] for product in products]
            else:
                # products: list of size N, where N is the number of models
                #           each element holds a list of products (products[*]['products'])
                selected_products = []
                products = products[:top_k]
                for j in range(top_k):
                    selected_products.append(products[j%len(products)]['products'][j])
            # Uniformly select a permutation
            permutation_sample = query_permutations[i]
            selected_products = [selected_products[index] for index in permutation_sample]
            query_info['permutation'] = permutation_sample
            query_info['products'] = selected_products
            dataset['queries'].append(query_info)

    # Store dataset
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate a dataset using several LLMs as paraphrasers')
    parser.add_argument('-i', '--input-queries-path', required=True, type=str, help='Path to a JSON-like array of queries')
    parser.add_argument('-s', '--search-engine-path', required=True, type=str, help='Path to the directory holding the search engine (generated by script create_index.py)')
    parser.add_argument('-o', '--output-path', required=True, type=str, help='Path to a directory where results will be stored')
    parser.add_argument('--output-name', required=True, type=str, help='Name of the final JSON dataset file name (do not include the extension .json)')
    parser.add_argument('--seed', type=int, default=5243534, help='Seed used for deterministic output')
    parser.add_argument('-n', '--samples-per-query', type=int, default=10, help='Number of samples for each query. Each sample will randomly shuffle the products associated to the query')
    parser.add_argument('-k', '--top-k', type=int, default=None, help='Number of products per query each model will paraphrase. If provided, the final dataset will chose a random model paraphrased product for each product position in the products list of each query')
    parser.add_argument('--share-permutations', action=argparse.BooleanOptionalAction, help='If enabled, share permutations across queries instead of sampling for each query')

    args = parser.parse_args()

    main(args)
