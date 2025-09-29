from recommender import generate_recommendation
from tools import transform_dataset, load_llm
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
from tqdm.auto import tqdm
import multiprocessing
import argparse
import json
import sys
import os

def process_dataset(dataset, llm_name, llm_host, results, results_queue, process_id, num_processes):

    # Load recommender LLM
    recommender_llm = load_llm(llm_name, base_url=llm_host)

    # Prompt used to ask the model for a recommendation
    prompt_template = dataset['prompt_template']

    # Round-robin parallelization
    for sample_pos in range(process_id, len(dataset['queries']), num_processes):

        if results['results'][sample_pos] is not None:
            sample_info = results['results'][sample_pos]
        else:
            query_info = dataset['queries'][sample_pos]

            response, parsed_response = generate_recommendation(
                recommender_llm = recommender_llm,
                titles          = [product['title'] for product in query_info['products']],
                descriptions    = [product['description'] for product in query_info['products']],
                query           = query_info['query'],
                prompt_template = prompt_template
            )

            sample_info = {
                'query': query_info['query'],
                'attack_pos': query_info['attack_pos'] if 'attack_pos' in query_info else None,
                'predicted_pos': parsed_response['article_number'] if parsed_response is not None else None,
                'response': response,
                'parsed_response': parsed_response
            }

        results_queue.put((sample_info, sample_pos))

def distribute_work(input_dataset_path, llm_name, hosts, output_path, store_freq=100):

    # Load evaluation dataset
    with open(input_dataset_path, 'r') as f:
        dataset = json.load(f)

    # Prepare results structure
    results = {
        'input_dataset': input_dataset_path,
        'recommender_llm': llm_name
    }

    if os.path.exists(output_path):
        # Load already generated results
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)

        # Integrity checks
        for key in results.keys():
            assert loaded_results[key] == results[key]

        results['results'] = loaded_results['results']
    else:
        results['results'] = [None] * len(dataset['queries'])

    # Split dataset and launch processes
    results_queue = multiprocessing.Queue()
    processes = []
    for i, llm_host in enumerate(hosts):
        process = multiprocessing.Process(target=process_dataset, args=(dataset, llm_name, llm_host, results, results_queue, i, len(hosts)))
        process.start()
        processes.append(process)

    # Collect all results
    for i in tqdm(range(len(dataset['queries'])), desc='Evaluation'):
        sample_info, sample_pos = results_queue.get()
        results['results'][sample_pos] = sample_info

        # Store generated results
        if ((i+1) % store_freq) == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=3)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=3)

    # Wait for all processes to finish (they should be already finished)
    for process in processes:
        process.join()

def main(args):

    load_dotenv()

    print("WARNING: Test the automatic checkpoin store")

    input_dataset_path = args.input_dataset_path
    llm_name = args.llm_name
    output_path = args.output_path
    hosts_path = args.hosts_path

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configure LLM recommenders
    if hosts_path is None:
        hosts = [None]
    else:
        with open(hosts_path, 'r') as f:
            hosts = list(map(lambda line: line.strip(), f.readlines()))

    # Launch processes
    distribute_work(input_dataset_path, llm_name, hosts, output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate responses from a dataset, used to evaluate the system')
    parser.add_argument('-i', '--input-dataset-path', required=True, type=str, help='Dataset path to be evaluated')
    parser.add_argument('-m', '--llm_name', required=True, type=str, help='Evaluated LLM name. Use Ollama format. If results path is included, this should not be specified')
    parser.add_argument('-o', '--output-path', required=True, type=str, help='Path to JSON-like file where results will be stored')
    parser.add_argument('--hosts-path', type=str, default=None, help='Path to a file of hosts used to do inference with Ollama models. The number of hosts will create the same number of parallel processes. If not provided, the default host will be used')

    args = parser.parse_args()

    main(args)
