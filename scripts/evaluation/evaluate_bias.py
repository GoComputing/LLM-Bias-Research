from recommender import generate_recommendation
from tools import transform_dataset
from langchain_ollama.llms import OllamaLLM
from tqdm.auto import tqdm
import argparse
import json
import os

def main(args):

    input_dataset_path = args.input_dataset_path
    results_path = args.results_path
    llm_name = args.llm_name
    output_path = args.output_path

    if os.path.exists(output_path):
        raise IOError(f'Output path already exists ({output_path})')

    if results_path is not None:
        if llm_name is not None:
            raise ValueError('If results path is provided, llm_name should not be provided')
        with open(results_path, 'r') as f:
            attack_results = json.load(f)
        llm_name = attack_results['metadata']['target_llm']
        paraphrase_prompt_template = attack_results['best_result']['prompt_template'] + \
            " Provide your answer in a JSON format. Use the format `{{\"paraphrased\": \"your answer\"}}, where \"your_answer\" is a single string`\n\n{text}"
    elif llm_name is None:
        raise ValueError('If no results path is specified, model name must be provided')

    # Configure LLM recommender
    recommender_llm = OllamaLLM(model=llm_name)
    recommender_llm.temperature = 0
    # recommender_llm.num_ctx = 131072    # TODO: this is for Llama3.1. Parametrize to be able to use other models
    recommender_llm.num_ctx     = 16384
    recommender_llm.num_predict = 4096  # Limit the generation

    if results_path is not None:
        recommender_llm.seed = attack_results['metadata']['seed']

    # Load evaluation dataset
    with open(input_dataset_path, 'r') as f:
        dataset = json.load(f)

    prompt_template = dataset['prompt_template']

    results = {
        'input_dataset': input_dataset_path,
        'attack_path': results_path, 
        'recommender_llm': llm_name,
        'results': []
    }

    if results_path is not None:
        dataset, transform_data = transform_dataset(dataset, recommender_llm, paraphrase_prompt_template, bar_pos=0)
        results['transform_data'] = transform_data

    for query_info in tqdm(dataset['queries'], desc='Evaluation'):

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
    parser.add_argument('-r', '--results-path', required=False, default=None, type=str, help='Results path from one attack. If not provided, baseline results will be computed')
    parser.add_argument('-m', '--llm_name', required=False, type=str, help='Evaluated LLM name. Use Ollama format. If results path is included, this should not be specified')
    parser.add_argument('-o', '--output-path', required=True, type=str, help='Path to JSON-like file where results will be stored')

    args = parser.parse_args()

    main(args)
