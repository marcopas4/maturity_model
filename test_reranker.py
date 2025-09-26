#!/usr/bin/env python3
"""
Simple test script to verify Jina reranker integration with the Agent class.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jina_reranker import JinaReranker
import config as cfg

def test_jina_reranker():
    """Test basic Jina reranker functionality."""
    print("Testing Jina Reranker integration...")
    
    try:
        # Initialize reranker
        reranker = JinaReranker(
            device='mps',
            use_fp16=cfg.JINA_USE_FP16,
            batch_size=cfg.JINA_BATCH_SIZE,
            max_length=cfg.JINA_MAX_LENGTH
        )
        print("✓ JinaReranker initialized successfully")
        
        # Test data
        query = "What are the security requirements for software development?"
        documents = [
            "This document outlines security requirements for software development including code review processes.",
            "The weather today is sunny with a temperature of 25 degrees celsius.",
            "Software security guidelines include input validation, authentication, and encryption requirements.",
            "This is a random text about cooking recipes and food preparation techniques."
        ]
        
        # Test reranking
        print(f"\nTesting reranking with query: '{query}'")
        print(f"Number of documents: {len(documents)}")
        
        result = reranker.rerank(
            query=query,
            documents=documents
        )
        
        print(f"✓ Reranking completed successfully")
        print(f"Number of results: {len(result.results)}")
        
        # Print results
        print("\nReranked results:")
        for i, res in enumerate(result.results):
            print(f"  {i+1}. Index: {res.index}, Score: {res.relevance_score:.4f}")
            print(f"     Text: {documents[res.index][:100]}...")
        
        # Get metrics
        metrics = reranker.get_metrics()
        print(f"\n✓ Metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_agent_integration():
    """Test that the Agent class can be imported without errors."""
    print("\nTesting Agent class integration...")
    
    try:
        # This will test if the imports work correctly
        from agent import Agent
        print("✓ Agent class imports successfully")
        
        # Note: We're not fully initializing the Agent here because it requires
        # documents and other resources that might not be available in this test
        
        return True
        
    except Exception as e:
        print(f"✗ Error importing Agent: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Jina Reranker Integration Test")
    print("=" * 60)
    
    success = True
    
    # Test reranker functionality
    success &= test_jina_reranker()
    
    # Test agent integration
    success &= test_agent_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Jina reranker integration is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)