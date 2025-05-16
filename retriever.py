from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
import faiss
import metadata_retriever as mtr
from typing import List


class Retriever:
    """
    A multi-strategy document retrieval system that implements different retrieval methods.

    This class provides three distinct retrieval strategies:
    1. Auto-merging Retrieval:
       - Hierarchical document retrieval
       - Automatic merging of related chunks
       - Uses vector similarity for initial retrieval
       - Simple ratio threshold for merging: 0.5
       
    2. BM25 Sparse Retrieval:
       - Keyword-based retrieval using BM25 algorithm
       - English stemming and stop words removal
       - Traditional information retrieval approach
       
    3. Metadata-based Retrieval:
       - Filtering based on document metadata
       - Semantic similarity with embeddings
       - Configurable similarity threshold
       - Returns top-k most relevant results

    Attributes:
        docstore: Document storage backend for managing text nodes
        nodes (List[TextNode]): Collection of input document nodes
        embed_model: HuggingFace embedding model from Settings
        vector_store (FaissVectorStore): FAISS vector store for efficient similarity search
        storage_context (StorageContext): Context managing document and vector stores
        leaf_nodes (List[TextNode]): Leaf nodes from document hierarchy
        root_nodes (List[TextNode]): Root nodes from document hierarchy
        vector_index (VectorStoreIndex): Vector index for similarity search
        base_retriever: Base vector retriever (similarity_top_k=3)
        auto_merging_retriever: Hierarchical retriever with smart merging
        sparse_retriever: BM25-based keyword retriever
        meta_retriever: Metadata-based semantic retriever

    Example:
        >>> from llama_index.core import SimpleDocumentStore
        >>> docstore = SimpleDocumentStore()
        >>> nodes = [TextNode(...), TextNode(...)]
        >>> retriever = Retriever(docstore, nodes)
        >>> results = retriever.retrieve(
        ...     query="What is machine learning?",
        ...     mode="auto-merging"
        ... )
    """

    def __init__(self, docstore, nodes: List[TextNode]) -> None:
        """
        Initialize the Retriever with document store and nodes.

        Args:
            docstore: Document storage backend
            nodes (List[TextNode]): List of document nodes to process

        Raises:
            ValueError: If nodes or docstore are empty/invalid
        """
        self.docstore = docstore
        self.nodes = nodes
        self.embed_model = Settings.embed_model 
        dimension = len(self.embed_model.get_query_embedding("test"))
        faiss_index = faiss.IndexFlatL2(dimension)
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
                
        # Creiamo un contesto di archiviazione con il vector store
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.docstore,
        )
                
        self.leaf_nodes = get_leaf_nodes(self.nodes)
        self.root_nodes = get_root_nodes(self.nodes)
        # Creazione dell'indice vettoriale
        self.vector_index = VectorStoreIndex(
            nodes=self.leaf_nodes,
            storage_context=self.storage_context,
        )
                
        # Otteniamo il retriever base
        self.base_retriever = self.vector_index.as_retriever(similarity_top_k=7)

           
            # Creiamo l'Auto Merging Retriever
        self.auto_merging_retriever = AutoMergingRetriever(
                simple_ratio_thresh=0.0,
                vector_retriever=self.base_retriever,
                storage_context=self.storage_context,
            )
        self.sparse_retriever = BM25Retriever(
                similarity_top_k=7,
                nodes=self.nodes,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
            )
        self.meta_retriever = mtr.MetadataRetriever(embedding_model=self.embed_model, similarity_threshold=0.5, top_k=5, nodes=self.leaf_nodes)
        self.meta_retriever.nodes = self.meta_retriever.create_nodes_with_metadata(self.meta_retriever.nodes)
                
        
    def retrieve(self, query: str, mode: str = "auto-merging") -> List[TextNode]:
        """
        Retrieve relevant documents using the specified retrieval strategy.

        Args:
            query (str): The search query string
            mode (str): Retrieval strategy to use. Options:
                - "auto-merging": Hierarchical retrieval with smart merging
                - "bm25": BM25 algorithm for keyword matching
                - "metadata": Metadata-based filtering with embeddings

        Returns:
            List[TextNode]: Retrieved document nodes, ordered by relevance

        Raises:
            ValueError: If retrieval fails or mode is invalid

        Example:
            >>> # Auto-merging retrieval
            >>> results = retriever.retrieve(
            ...     "What is machine learning?",
            ...     mode="auto-merging"
            ... )
            >>> 
            >>> # BM25 retrieval
            >>> results = retriever.retrieve(
            ...     "gradient descent algorithm",
            ...     mode="bm25"
            ... )
            >>> 
            >>> # Metadata retrieval
            >>> results = retriever.retrieve(
            ...     "neural networks",
            ...     mode="metadata"
            ... )
        """
        try:
            if mode == "metadata":
                self.nodes = self.meta_retriever.filter_nodes(query)
                return self.nodes
            if mode == "bm25":
                return self.sparse_retriever.retrieve(query)
            if mode == "auto-merging":
                return self.auto_merging_retriever.retrieve(query)
                        
        except Exception as e:
            raise ValueError(f"Retrieval failed: {str(e)}")




