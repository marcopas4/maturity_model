from llama_index.core.schema import TextNode
import numpy as np
from typing import List, Dict, Any
from llama_index.core import Settings
from langchain_ollama import ChatOllama
import json,os
from llama_index.core.schema import Node
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes,get_root_nodes
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

class MetadataRetriever:
    """
    A class that implements metadata-based retrieval and filtering for text nodes.

    This class provides functionality to:
    - Extract keywords from text chunks using LLM
    - Store and retrieve nodes with metadata
    - Filter nodes based on semantic similarity between query and keywords
    - Cache processed nodes to avoid recomputation

    Attributes:
        embedding_model: Model used for text embeddings
        similarity_threshold (float): Minimum similarity score for filtering (0-1)
        top_k (int): Number of top results to return
        nodes (List[TextNode]): Collection of text nodes to process
    """

    def __init__(self, embedding_model, similarity_threshold=0.5, top_k=3, nodes=None):
        """
        Initialize the MetadataRetriever.

        Args:
            embedding_model: Model for computing text embeddings
            similarity_threshold (float): Minimum similarity score (default: 0.5)
            top_k (int): Number of results to return (default: 3)
            nodes (List[TextNode]): Input nodes to process
        
        Raises:
            ValueError: If nodes parameter is None
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.nodes = nodes
        '''if not self.nodes:
            raise ValueError("Nodes not provided")
        '''
        
        
    def keyword_extraction(self,text:str):
        """
        Extract keywords from text using LLM in JSON format.

        Args:
            text (str): Input text to extract keywords from

        Returns:
            Response: LLM response containing keywords in JSON format
            {"keywords": ["keyword1", "keyword2", "keyword3"]}
        """
        local_llm = "llama3.2:latest"
        llm_json_mode = ChatOllama(model=local_llm,temperature=0, format="json")
        prompt = f"""Estrai tre keywords da questo testo, cercando di mantenere il significato originale, 
        e dammele in output in formato JSON.
        Le keyword vanno date in output prendendo ad esempio lo schema di questo esempio:
        {{'keywords': ['keyword 1', 'keyword 2', 'keyword 3']}}.
        
        Ricorda, ciò che hai letto è un esempio e non un output fisso da dare.
        Qui di seguito trovi il testo da cui devi estrarle: 
        {text}
        """

        response = llm_json_mode.invoke(prompt)
        return response
    
    def create_nodes_with_metadata(self,nodes: List[TextNode]) -> List[TextNode]:
        """
        Process nodes to add keyword metadata and handle caching.

        Args:
            nodes (List[TextNode]): Input nodes to process

        Returns:
            List[TextNode]: Processed nodes with keyword metadata

        Note:
            Nodes are cached in JSON format to avoid reprocessing
        """
        
        file_path = os.path.join("data", "nodes_data.json")
        if os.path.isfile(file_path):
            leafs = self.load_nodes()
            for idx,node in enumerate(leafs):
                nodes[idx].metadata = node.metadata
                #print(nodes[idx].metadata)
        else:

            for node in nodes:
                chunk_text = node.get_content()
                keywords = self.get_keywords_from_chunk(chunk_text)
                node.metadata["keywords"] = keywords
                print(node.metadata)
            self.save_nodes(nodes)
        return nodes
    
    def get_keywords_from_chunk(self,chunk_text: str) -> List[str]:
        """
        Extract keywords from a text chunk and parse JSON response.

        Args:
            chunk_text (str): Text chunk to process

        Returns:
            List[str]: List of extracted keywords
        """
        response = self.keyword_extraction(chunk_text)
        try:
            data = json.loads(response.content)
            
            if isinstance(data, dict) and "keywords" in data:
                return data["keywords"]
            return []
        except json.JSONDecodeError:
            return []

    def load_nodes(self) -> List[TextNode]:
        """
        Load cached nodes from JSON file.

        Returns:
            List[TextNode]: List of nodes reconstructed from JSON

        Note:
            Looks for nodes_data.json in the data directory
        """
        file_path = os.path.join("data", "nodes_data.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            nodes_dict = json.load(f)
            nodes = [Node.from_dict(node_data) for node_data in nodes_dict]
            return nodes

    def save_nodes(self,nodes: List[TextNode]):
        """
        Save nodes to JSON file for caching.

        Args:
            nodes (List[TextNode]): Nodes to serialize and save
        """
        file_path = os.path.join("data", "nodes_data.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            nodes_dict = [node.to_dict() for node in nodes]
            json.dump(nodes_dict, f, ensure_ascii=False, indent=2)

    def filter_nodes(self, query: str) -> List[tuple]:
        """
        Filter nodes based on semantic similarity between query and keywords.

        Args:
            query (str): Query string to match against node keywords

        Returns:
            List[tuple]: List of (node, similarity_score) tuples, sorted by score
                        Limited to top_k results

        Note:
            Uses cosine similarity between query embedding and keyword embeddings
        """
        
        query_embedding = self.embedding_model.get_text_embedding(query)
        
        filtered_nodes = []
        for node in self.nodes:
            keywords = node.metadata.get("keywords", "")
            if not keywords:
                continue
                
            if isinstance(keywords, str):
                keywords_list = [k.strip() for k in keywords.split(",")]
            else:
                keywords_list = keywords
                
            keywords_text = " ".join(keywords_list)
            keywords_embedding = self.embedding_model.get_text_embedding(keywords_text)
            
            similarity = np.dot(query_embedding, keywords_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(keywords_embedding)
            )
            
            if similarity >= self.similarity_threshold:
                filtered_nodes.append((node,similarity))

        filtered_nodes = sorted(filtered_nodes, key=lambda x: x[1], reverse=True)
                
        return filtered_nodes[:self.top_k]
    
    def display(self, filtered_nodes: List[tuple]):
        """
        Display filtered nodes with their metadata and scores.

        Args:
            filtered_nodes (List[tuple]): List of (node, score) tuples to display

        Prints:
            Node ID, similarity score, metadata, and content for each result
        """
        for node in filtered_nodes:
            print(f"Node ID: {node[0].node_id}, Score: {node[1]},  Metadata: {node[0].metadata} \n Content: {node[0].get_content()} \n\n")


