from metadata_retriever import MetadataRetriever
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
)
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
import utils.document_loader as dl
import config as cfg

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_model = HuggingFaceEmbedding(
        model_name="all-MiniLM-L6-v2",
        device=device
    )
Settings.embed_model = embed_model

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 1024, 2048],
    chunk_overlap=100, 
       
)
docs = dl.load_documents_from_directory(cfg.DOCUMENTS_DIR)
nodes = node_parser.get_nodes_from_documents(docs)


leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)

    # Initialize MetadataRetriever
retriever = MetadataRetriever(
        embedding_model=embed_model,
        similarity_threshold=0.5,
        top_k=2,
        nodes=leaf_nodes
    )

    # Test queries
test_queries = [
        "È stato sviluppato e implementato un programma di sicurezza delle informazioni (compresi i programmi di tutti i settori rilevanti della matrice di controllo del cloud)?",
        "Il programma di governance prevede un processo di eccezione approvato e seguito ogni volta che si verifica una deviazione da una politica stabilita?",
        "Tutte le politiche organizzative pertinenti e le procedure associate sono riviste almeno annualmente?"
    ]

    # Run tests
print("Testing MetadataRetriever...")
print("-" * 50)

retriever.nodes = retriever.create_nodes_with_metadata(retriever.nodes)

for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        # Get filtered nodes
        results = retriever.filter_nodes(query)
        
        # Display results
        retriever.display(results)