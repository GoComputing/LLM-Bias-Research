from tools import extract_all_json, build_enhancer_prompt_template, paraphrase_text
from recommender import generate_recommendation
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import argparse
import json
import os


def evaluate_dataset(dataset, target_llm):

    prompt_template = dataset['prompt_template']
    queries = dataset['queries']
    evaluation_data = []

    # Iterate dataset
    for query_info in tqdm(queries, desc='Current evaluation', position=1):

        # Get single sample information
        query = query_info['query']
        titles = [product['title'] for product in query_info['products']]
        descriptions = [product['description'] for product in query_info['products']]

        # Evaluate single sample
        response, parsed_response = generate_recommendation(target_llm, titles, descriptions, query, prompt_template=prompt_template)

        print(parsed_response)

        evaluation_data.append({
            'predicted_pos': parsed_response['article_number'] if parsed_response is not None else -1,
            'response': response,
            'parsed_response': parsed_response
        })

    target_pos = [query_info['attack_pos'] for query_info in queries]
    predicted_pos = [sample_info['predicted_pos'] for sample_info in evaluation_data]

    # Compute metric (TODO: allow selecting metric)
    accuracy = accuracy_score(target_pos, predicted_pos)

    return accuracy, evaluation_data

    


def launch_tree_attack_rec(dataset, attacker_llm, target_llm, full_tree, current_branch, evaluations, max_evaluations, max_depth, num_childs, output_dir, progress_bar, parent_score, prompt_template):

    # TODO: Transform dataset

    # Node evaluation
    current_eval = len(evaluations)
    metric_value, evaluation_data = evaluate_dataset(dataset, target_llm)
    evaluations.append({'metric_value': metric_value, 'prompt_template': prompt_template, 'data': evaluation_data})
    progress_bar.update(1)

    evaluation_path = os.path.join(output_dir, 'evaluations', f'{current_eval:05}.json')
    with open(evaluation_path, 'w') as f:
        json.dump(evaluations[-1])

    # Generate child nodes
    if current_eval >= max_evaluations or max_depth == 0 or metric_value < parent_score:
        num_skip_steps = ((num_childs ** (max_depth+1) - 1) // (num_childs-1)) - 1
        progress_bar.update(num_skip_steps)
        current_branch[current_eval] = None
    else:
        # Transform prompt
        # TODO: key idea is to take all prompts and scores (ancestors). Then, paraphrase the current prompt using this information
        #       variable holding prompt is prompt_template
        
        current_branch[current_eval] = dict()
        for i in range(num_childs):
            launch_tree_attack_rec(dataset, attacker_llm, target_llm, full_tree, current_branch[current_eval], evaluations, max_evaluations, max_depth-1, num_childs, output_dir, progress_bar)


def launch_tree_attack(dataset, attacker_llm, target_llm, max_evaluations, max_depth, num_childs, output_dir):

    # Adjust max_evaluations
    max_theoretical_evals = (num_childs ** (max_depth+1) - 1) // (num_childs-1)
    if max_evaluations > max_theoretical_evals:
        max_evaluations = max_theoretical_evals

    metadata = {
        'max_depth': max_depth,
        'max_evaluations': max_evaluations,
    }

    progress_bar = tqdm(total=max_evaluations, position=0, desc='Evaluations')
    evaluations = []

    # Set an initial prompt template for paraphrasing a description
    prompt_template  = "This is the product description from an online tool. Enhance the description so the likely of being recommendated is increased. Do not change the meaning of the original text. Provide your answer in a JSON format. Use the format `{{\"paraphrased\": \"your answer\"}}`\n"
    prompt_template += "\n"
    prompt_template += "{text}"

    full_tree = dict()
    launch_tree_attack_rec(
        dataset, attacker_llm, target_llm,
        full_tree, full_tree, evaluations,
        max_evaluations, max_depth, num_childs, output_dir,
        progress_bar,
        parent_score=float('-inf'), dataset['prompt_template']
    )

    tree_path = os.path.join(output_dir, 'tree.json')
    with open(tree_path, 'w') as f:
        json.dump(full_tree, f, indent=3)

    # TODO: store results (best prompt, metadata, ...)
    #       best prompt can be easily retrieved from 'evaluations' list, where there is a field with the score and the prompt


def main(args):

    dataset_path = args.dataset_path
    model_name = args.model_name
    output_dir = args.output_path
    max_depth = args.max_depth
    max_evaluations = args.max_evals
    num_childs = args.num_childs

    if not os.path.exists(output_dir):
        raise IOError(f'Output directory does not exists ({output_dir})')

    if not os.path.isdir(output_dir):
        raise IOError(f'Output path is not a directory ({output_dir})')

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Reorder keys
    queries = dataset['queries']
    del dataset['queries']
    dataset['dataset_path'] = dataset_path
    dataset['attacker_model'] = model_name
    dataset['target_model'] = model_name
    dataset['eval_model'] = model_name
    dataset['queries'] = queries

    # Configure LLMs
    attacker_llm = OllamaLLM(model=model_name)
    attacker_llm.temperature = 0
    attacker_llm.num_ctx     = 16384
    attacker_llm.num_predict = 4096  # Limit the generation

    target_llm = attacker_llm

    # Launch attack
    results = launch_tree_attack(dataset, attacker_llm, target_llm, max_evaluations, max_depth, num_childs, output_dir)

    # Evaluate dataset
    # TODO


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Implementation of tree attack')
    parser.add_argument('-d', '--dataset-path', required=True, type=str, help='Dataset to be attacked. It can be generated by the script generate_queries.py')
    parser.add_argument('-m', '--model-name', required=True, type=str, help='Attacker, target and evaluation model (all are the same). Use the Ollama format')
    parser.add_argument('-o', '--output-path', required=True, type=str, help='Directory where results will be stored')
    parser.add_argument('--max-depth', default=4, type=int, help='Maximum tree depth to be explored')
    parser.add_argument('--max-evals', default=500, type=int, help='Maximum number of dataset evaluations')
    parser.add_argument('--num-childs', default=3, type=int, help='Number of childs for each node to be generated')

    args = parser.parse_args()

    main(args)
