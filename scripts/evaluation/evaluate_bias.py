from recommender import generate_recommendation
from langchain_ollama.llms import OllamaLLM
from tqdm.auto import tqdm
import argparse
import json
import os

def main(args):

    input_dataset_path = args.input_dataset_path
    llm_name = args.llm_name
    output_path = args.output_path

    if os.path.exists(output_path):
        raise IOError(f'Output path already exists ({output_path})')

    # Configure LLM recommender
    recommender_llm = OllamaLLM(model=llm_name)
    recommender_llm.temperature = 0
    # recommender_llm.num_ctx = 131072    # TODO: this is for Llama3.1. Parametrize to be able to use other models
    recommender_llm.num_ctx     = 8192
    recommender_llm.num_predict = 2000  # Limit the generation to 2000 tokens

    # Load evaluation dataset
    with open(input_dataset_path, 'r') as f:
        dataset = json.load(f)

    prompt_template = dataset['prompt_template']

    results = {
        'input_dataset': input_dataset_path,
        'recommender_llm': llm_name,
        'results': []
    }

    for query_info in dataset['queries']:

        response, parsed_response = generate_recommendation(
            recommender_llm = recommender_llm,
            titles          = [product['title'] for product in query_info['products']],
            descriptions    = [product['description'] for product in query_info['products']],
            query           = query_info['query'],
            prompt_template = prompt_template
        )

        results['results'].append({
            'query': query_info['query'],
            'attack_pos': query_info['attack_pos'],
            'predicted_pos': parsed_response['article_number'] if parsed_response is not None else None,
            'response': response,
            'parsed_response': parsed_response
        })

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=3)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate responses from a dataset, used to evaluate the system')
    parser.add_argument('-i', '--input-dataset-path', required=True, type=str, help='Dataset path to be evaluated')
    parser.add_argument('-m', '--llm_name', required=True, type=str, help='Evaluated LLM name. Use Ollama format')
    parser.add_argument('-o', '--output-path', required=True, type=str, help='Path to JSON-like file where results will be stored')

    args = parser.parse_args()

    main(args)
