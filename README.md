# LLM Bias Research


Start LLM server

docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama


Available models

https://ollama.com/library


Download a model

docker exec -it ollama ollama pull <model-name>[:<tag>]


Launch a script from this repository

export PYTHONPATH="$(pwd)/src"


Generate index

python scripts/create_dataset.py -d data/dataset/train.csv -o data/amazon_index