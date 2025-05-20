from tools import paraphrase_text, load_llm
from tqdm.auto import tqdm
from copy import deepcopy
import argparse
import random
import json
import os

def main(args):

    input_path = args.input_path
    output_path = args.output_path
    n_samples_per_query = args.n_samples_per_query
    model_name = args.model_name
    temperature = args.temperature
    seed = args.seed

    if os.path.exists(output_path):
        raise IOError(f'Output directory already exists ({output_path})')

    with open(input_path, 'r') as f:
        dataset = json.load(f)

    # Load model
    paraphraser_llm = load_llm(model_name)

    original_queries = dataset['queries']
    del dataset['queries']

    dataset['dataset_path'] = input_path
    dataset['n_samples_per_query'] = n_samples_per_query
    dataset['model_name'] = model_name
    dataset['temperature'] = temperature
    dataset['seed'] = seed

    dataset['queries'] = []

    for query_info in tqdm(original_queries, desc='Queries', position=0):

        query = query_info['query']

        for i in tqdm(range(n_samples_per_query), desc='Samples', position=1):

            random.seed(seed)
            paraphraser_llm.seed = seed
            seed = random.randint(0, 2**32-1)

            new_query, paraphrase_data = paraphrase_text(
                paraphraser_llm,
                query,
                return_raw_response=True, original_on_failure=True)

            new_query_info = deepcopy(query_info)
            new_query_info['query'] = new_query
            new_query_info['paraphrase_data'] = paraphrase_data

            dataset['queries'].append(new_query_info)

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=3)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to sample queries from a seed dataset')
    parser.add_argument('-i', '--input-path', required=True, type=str, help='Queries dataset path')
    parser.add_argument('-o', '--output-path', required=True, type=str, help='Output path as a JSON-like file')
    parser.add_argument('-n', '--n-samples-per-query', required=True, type=int, help='Number of samples for each query')
    parser.add_argument('-m', '--model-name', required=True, type=str, help='LLM model name used to paraphrase. Use Ollama format')
    parser.add_argument('-t', '--temperature', default=1, type=float, help='Temperature used to sample queries from a seed query')
    parser.add_argument('--seed', default=753245, type=int, help='Seed used for reproducibility purposes')

    args = parser.parse_args()

    main(args)
