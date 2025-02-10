from searchengine import AmazonSearchEngine
import argparse
import os

def main(args):

    # Get arguments
    amazon_data_path = args.data_path
    database_max_nsamples = args.nsamples
    seed = args.seed
    amazon_index_path = args.output_path

    if os.path.exists(amazon_index_path):
        raise IOError('Output directory already exists')

    # Generate and serialize
    search_engine = AmazonSearchEngine()

    print("Initializing search engine...")
    search_engine.initialize_search_engine(amazon_data_path, database_max_nsamples, seed)
    print("Store search engine...")
    search_engine.serialize(amazon_index_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to generate vector store index and store it')
    parser.add_argument('-d', '--data-path', type=str, required=True, help='Amazon CSV dataset path. It can be downloaded from https://www.kaggle.com/datasets/piyushjain16/amazon-product-data')
    parser.add_argument('-s', '--seed', type=int, default=64562, help='Seed used to sample the dataset')
    parser.add_argument('-n', '--nsamples', type=int, default=50000, help='Dataset size after sampling')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='Directory used to store the VectorStoreIndex')

    args = parser.parse_args()

    main(args)
