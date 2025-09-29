# LLM Bias Research


Configure Ollama
 1. Start server
    $ docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 2. Select a model
    https://ollama.com/library
 3. Download a model
    $ docker exec -it ollama ollama pull <model-name>[:<tag>]


Download required data
 1. Download initial dataset (train.csv)
    https://www.kaggle.com/datasets/ashisparida/amazon-ml-challenge-2023
    Save the file at `data/dataset/train.csv`


General steps

 1. Configure environment
    $ conda env create -n syntactic-bias -f environment.yml
    $ conda activate syntactic-bias
    $ export PYTHONPATH="$(pwd)/src"
 2. Generate search engine
    $ python scripts/dataset_creation/create_index.py -d data/dataset/train.csv -o data/amazon_index
 3. Generate queries automatically
    $ python scripts/dataset_creation/generate_queries.py -o data/generated_queries.json
 4. Sample paraphrased products
    $ python scripts/dataset_creation/paraphrase_products.py -i data/generated_queries.json -s data/amazon_index/ -o data/paraphrase_dataset --output-name model_bias_dataset -n 30 -k 5 --share-permutations
 5. Generate evaluation
    $ python scripts/evaluation/evaluate_bias.py -i data/paraphrase_dataset/model_bias_dataset.json -m <TARGET_MODEL> -o results/main_results/evaluation_<MODEL_NAME>.json