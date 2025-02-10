from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Document, VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pandas as pd
import json
import os


class AmazonSearchEngine:

    def _load_amazon_csv(self):
        """
        Loads CSV database from `self.metadata['data_path']` and sample it
        """

        # Load Amazon product descriptions
        self.amazon_data = pd.read_csv(self.metadata['data_path'])

        # Preprocess data
        self.amazon_data = self.amazon_data.dropna(subset=['DESCRIPTION', 'PRODUCT_TYPE_ID'])
        if len(self.amazon_data) > self.metadata['max_nsamples']:
            self.amazon_data = self.amazon_data.sample(self.metadata['max_nsamples'], replace=False, random_state=self.metadata['seed'])


    def __init__(self, embedding_model='BAAI/bge-small-en-v1.5'):
        """
        Initializes the search engine with a specific embedding model

        Parameters:
          embedding_model (str): Embedding model name (HuggingFace standard)
        """

        self.embedding_model_name = embedding_model
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)


    def initialize_search_engine(self, data_path, max_nsamples, seed):
        """
        Generates metadata and creates a vector store index from llama_index package

        Parameters:
          data_path (str): Path of the Amazon CSV dataset
          seed (int): Seed used to sample the Amazon dataset
          max_nsamples (int): Maximum number of samples. Typically the desired sampled dataset size
        """

        self.metadata = {
            'data_path': data_path,
            'seed': seed,
            'max_nsamples': max_nsamples,
            'embedding_model': self.embedding_model_name
        }

        self._load_amazon_csv()

        # Generate embeddings from titles
        amazon_documents = [Document(text=row['TITLE'], metadata={'ID': row['PRODUCT_ID']}) for row_id, row in self.amazon_data.iterrows()]
        self.amazon_index = VectorStoreIndex.from_documents(amazon_documents, show_progress=True, insert_batch_size=len(amazon_documents))
    

    def serialize(self, store_dir):
        """
        Stores the vector store index and metadata

        Parameters:
          store_dir (str): Directory where data will be stored
        """

        if os.path.exists(store_dir):
            raise IOError(f'Path already exists {store_dir}')

        # Store vector index (embeddings)
        vector_index_path = os.path.join(store_dir, 'amazon_index')
        self.amazon_index.storage_context.persist(persist_dir=vector_index_path)

        # Store metadata
        metadata_path = os.path.join(store_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=3)


    def load_search_engine(self, load_dir):
        """
        Loads the vector store index stored using `self.serialize`

        Parameters:
          load_dir (str): Directory where data is stored. Use the same directory as the used for `self.serialize`
        """

        # Load metadata
        metadata_path = os.path.join(load_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self._load_amazon_csv()

        # Load embeddings
        amazon_index_path = os.path.join(load_dir, 'amazon_index')
        storage_context = StorageContext.from_defaults(persist_dir=amazon_index_path)
        self.amazon_index = load_index_from_storage(storage_context)


    def query(self, text, top_k=10):
        """
        Retrieves the closest top_k samples from the vector store index

        Parameters:
          text (str): query to semantically match the dataset samples
        """

        amazon_retriever = self.amazon_index.as_retriever(similarity_top_k=top_k)
        responses = amazon_retriever.retrieve(text)
        products_ids = [response.metadata['ID'] for response in responses]
        df = self.amazon_data[self.amazon_data['PRODUCT_ID'].isin(products_ids)]
        return df
