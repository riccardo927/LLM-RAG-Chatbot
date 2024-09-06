from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

class DocumentEmbedder:
    def __init__(self, embedding_type, credentials=None, region_name=None, api_key=None):
        """
        Initializes the embedder class based on the selected embedding type.
        
        :param embedding_type: The type of embedding model to use. Options are: 
                               'bedrock', 'ollama_mxbai', 'huggingface', 'ollama_nomic', 'titan'
        :param credentials: AWS credentials for Bedrock embeddings (optional).
        :param region_name: AWS region for Bedrock embeddings (optional).
        :param api_key: API key for HuggingFace Inference API embeddings (optional).
        """
        self.embedding_type = embedding_type
        self.credentials = credentials
        self.region_name = region_name
        self.api_key = api_key

        # Select embedding method based on the type
        self.embeddings = self.select_embedding_function()

    def select_embedding_function(self):
        """
        Selects the embedding function based on the embedding type.
        """
        if self.embedding_type == "bedrock":
            return BedrockEmbeddings(credentials_profile_name=self.credentials, region_name=self.region_name)
        
        elif self.embedding_type == "ollama_mxbai":
            return OllamaEmbeddings(model="mxbai-embed-large")
        
        elif self.embedding_type == "Titan-text-embeddings-v2":
            return HuggingFaceInferenceAPIEmbeddings(
                api_key=self.api_key, 
                model_name="amazon/Titan-text-embeddings-v2"
            )
        
        elif self.embedding_type == "ollama_nomic":
            return OllamaEmbeddings(model="nomic-embed-text")
        
        elif self.embedding_type == "sentence-transformers":
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
        
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

    def embed_documents(self, documents):
        """
        Embeds the documents using the selected embedding model.
        """
        return [self.embeddings.embed(doc) for doc in documents]

    def load_chroma(self, db_path):
        """
        Loads the Chroma vector store using the embedding function.
        """
        return Chroma(persist_directory=db_path, embedding_function=self.embeddings)

