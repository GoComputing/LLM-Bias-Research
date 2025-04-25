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
    $ conda create -n llm_bias_attack -f environment.yml
    $ conda activate llm_bias_attack
    $ export PYTHONPATH="$(pwd)/src"
 2. Generate search engine
    $ python scripts/dataset_creation/create_index.py -d data/dataset/train.csv -o data/amazon_index
 3. Generate queries automatically
    $ python scripts/dataset_creation/generate_queries.py -o data/generated_queries.json
 4. Create train and evaluation datasets
    $ python scripts/dataset_creation/create_dataset.py -q data/curated_queries.json -s data/amazon_index -o data/train_bias_dataset.json -k 5
    $ python scripts/dataset_creation/create_dataset.py -q data/generated_queries.json -s data/amazon_index -o data/eval_bias_dataset.json -k 5
 5. Launch some of the implemented attacks
    5.1 Paraphraser attack
        $ python scripts/attacks/paraphraser.py -d data/eval_bias_dataset.json -o results/self_bias_analysis/<ATTACK>/queries.json -m <ATTACKER_MODEL>
 6. Generate evaluation
    6.1 Generate baseline results
        $ python scripts/evaluation/evaluate_bias.py -i data/eval_bias_dataset.json -m <TARGET_MODEL> -o results/self_bias_analysis/baseline__no_attack/evaluation.json
    6.2 Generate attack results
        $ python scripts/evaluation/evaluate_bias.py -i results/self_bias_analysis/<ATTACK>/queries.json -m <TARGET_MODEL> -o results/self_bias_analysis/<ATTACK>/evaluation.json