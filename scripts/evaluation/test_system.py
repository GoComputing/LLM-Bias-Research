from recommender import RecommendationSystem, base_prompt_template
from searchengine import AmazonSearchEngine
from langchain_ollama.llms import OllamaLLM
from tqdm.auto import tqdm
import argparse
import json
import os

def main(args):

    amazon_index_path = args.index_dir
    queries_path = args.queries_path
    llm_name = args.model
    output_path = args.output_path
    initial_seed = args.seed
    repeats = args.repeats

    if not os.path.exists(os.path.dirname(output_path)):
        raise IOError('Output directory does not exists')

    if os.path.isdir(output_path):
        raise IOError('Output should be a path to a JSON file, not a directory')

    # Load queries
    with open(queries_path, 'r') as f:
        queries = json.load(f)

    # Load search engine
    print("Loading index...")
    search_engine = AmazonSearchEngine()
    search_engine.load_search_engine(amazon_index_path)

    # Initialize recommender system
    recommender_llm = OllamaLLM(model=llm_name)
    recommender_llm.temperature = 0
    recommender_llm.num_ctx = 131072
    recommender_llm.num_predict = 2000
    recommendation_system = RecommendationSystem(search_engine, recommender_llm, top_k=5, shuffle=True, initial_seed=initial_seed)

    res = {
        'prompt_template': base_prompt_template(),
        'llm_model_name': llm_name,
        'embedding_model_name': search_engine.embedding_model_name,
        'temperature': f'{recommender_llm.temperature:.5f}',
        'initial_seed': initial_seed,
        'max_tokens': recommender_llm.num_predict
    }

    # Return from checkpoint
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)

            # Check for consistency
            for attribute in res.keys():
                assert data[attribute] == res[attribute]
            assert 'results' in data
            assert 'last_seed' in data

            res = data
    else:
        res['results'] = []

    if 'last_seed' in res:
        recommendation_system.seed = res['last_seed']

    # Generate responses
    queries_iterator = tqdm(enumerate(queries), position=0, total=len(queries))
    for query_pos, query in queries_iterator:
        res_query = []

        # Check if already generated
        if query_pos < len(res['results']):
            continue
        
        for i in tqdm(range(repeats), position=1):
            
            matches, response, parsed_response = recommendation_system.query(query)
            res_query.append({
                "matches": [{
                    "title": row['TITLE'],
                    "description": row['DESCRIPTION']
                } for row_id, row in matches.iterrows()],
                "response": response,
                "parsed_response": parsed_response
            })

        # Store results
        res['last_seed'] = recommendation_system.seed
        res['results'].append(res_query)
        with open(output_path, 'w') as f:
            json.dump(res, f, indent=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test the implemented recommendation system')
    parser.add_argument('-i', '--index-dir', type=str, required=True, help='Path to the store index generated by `create_dataset.py`')
    parser.add_argument('-q', '--queries-path', type=str, required=True, help='A path to a JSON-like file which holds a list of queries')
    parser.add_argument('-m', '--model', type=str, default='llama3.1', help='LLM model name. This name should be available in ollama')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='A path to a non-existing JSON file where results will be stored')
    parser.add_argument('-s', '--seed', type=int, default=63456, help='Seed used for deterministic output')
    parser.add_argument('-n', '--repeats', type=int, default=1, help='Number of responses to generate for each query')

    args = parser.parse_args()

    main(args)
