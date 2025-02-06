# LLM Bias Research


Start LLM server

docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama


Available models

https://ollama.com/library


Download a model

docker exec -it ollama ollama pull <model-name>[:<tag>]
