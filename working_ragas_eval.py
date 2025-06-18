#!/usr/bin/env python3
"""
Working Ragas evaluation that uses the same embedding function as the existing ChromaDB.
"""

import os
import json
import chromadb
import google.generativeai as genai
from typing import List, Tuple
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall, 
    faithfulness,
    answer_relevancy
)
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from chromadb import Documents, EmbeddingFunction, Embeddings as ChromaEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

class GoogleEmbeddingFunction(EmbeddingFunction):
    """Google embedding function matching the one used in ChromaDB."""
    def __call__(self, input: Documents) -> ChromaEmbeddings:
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]
            
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model='embedding-001',
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.append(result['embedding'])
            
        return embeddings

class LangChainGoogleEmbeddings(Embeddings):
    """LangChain-compatible Google embeddings for Ragas."""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=self.api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model='embedding-001',
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.append(result['embedding'])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model='embedding-001',
            content=text,
            task_type="RETRIEVAL_QUERY"
        )
        return result['embedding']

class WorkingQTLEvaluator:
    """Working QTL RAG evaluator using correct embedding function."""
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in config.env")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in config.env")
        
        # Configure Google API
        genai.configure(api_key=self.google_api_key)
        
        # Initialize LLM for evaluation
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Using smaller, faster model for testing
            api_key=self.openai_api_key,
            temperature=0
        )
        
        # Initialize Google embeddings to match ChromaDB
        self.embeddings = LangChainGoogleEmbeddings()
        
        # Initialize ChromaDB client with correct embedding function
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
            # Get collection with the same embedding function
            self.collection = self.chroma_client.get_collection(
                "qtl_database",
                embedding_function=GoogleEmbeddingFunction()
            )
            print("✅ Connected to existing ChromaDB collection with correct embedding function")
        except Exception as e:
            print(f"❌ Could not load existing collection: {e}")
            self.collection = None
    
    def query_rag_system(self, question: str, n_results: int = 3) -> Tuple[str, List[str]]:
        """Query the RAG system and return answer and retrieved contexts."""
        if not self.collection:
            return "Error: Collection not available", []
        
        # Retrieve relevant contexts
        query_results = self.collection.query(
            query_texts=[question],
            n_results=n_results,
            include=['documents', 'distances']
        )
        
        contexts = query_results['documents'][0] if query_results['documents'] else []
        
        # Generate answer using retrieved contexts
        context_text = "\n\n".join(contexts[:3])  # Use top 3 contexts
        
        prompt = f"""Based on the following context from a QTL (Quantitative Trait Loci) genetics dataset, please answer the question.

Context:
{context_text}

Question: {question}

Please provide a comprehensive answer based on the provided context. If the context doesn't contain enough information to fully answer the question, please state that clearly.

Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        return answer, contexts
    
    def create_test_dataset(self) -> List[dict]:
        """Create a test dataset with QTL-specific questions."""
        return [
            {
                "question": "What is a LOD score and how is it interpreted?",
                "ground_truth": "A LOD score is the logarithm of odds that measures evidence for genetic linkage. A LOD score above 3 indicates significant evidence for linkage, with higher scores indicating stronger evidence."
            },
            {
                "question": "What is the difference between cis-acting and trans-acting QTLs?",
                "ground_truth": "Cis-acting QTLs are located near the gene they regulate (typically within 10 Mb), suggesting local genetic effects. Trans-acting QTLs are located far from the regulated gene, indicating distant regulatory mechanisms."
            },
            {
                "question": "What tissue was studied in this QTL analysis?",
                "ground_truth": "This QTL analysis was performed on liver tissue from Diversity Outbred mice."
            },
            {
                "question": "What do high LOD scores indicate about genetic regulation?",
                "ground_truth": "High LOD scores indicate strong evidence for genetic regulation, suggesting that genetic variants have substantial effects on gene expression levels."
            }
        ]
    
    def run_evaluation(self):
        """Run the complete Ragas evaluation."""
        print("🔬 Running Working Ragas Evaluation")
        print("=" * 40)
        
        # Create test dataset
        test_questions = self.create_test_dataset()
        
        # Prepare data for evaluation
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        print(f"📝 Processing {len(test_questions)} test questions...")
        
        for i, test_q in enumerate(test_questions):
            print(f"   Processing question {i+1}: {test_q['question'][:50]}...")
            
            question = test_q["question"]
            ground_truth = test_q["ground_truth"]
            
            # Get answer and contexts from RAG system
            answer, retrieved_contexts = self.query_rag_system(question)
            
            questions.append(question)
            answers.append(answer)
            contexts.append(retrieved_contexts)
            ground_truths.append(ground_truth)
            
            print(f"     Answer: {answer[:100]}...")
            print(f"     Retrieved {len(retrieved_contexts)} contexts")
        
        print(f"✅ Prepared {len(questions)} questions for evaluation")
        
        # Create dataset for Ragas
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })
        
        print("🔬 Running Ragas evaluation metrics...")
        
        # Run evaluation
        try:
            result = evaluate(
                eval_dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy
                ],
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            print("✅ Evaluation completed successfully!")
            
            # Print results
            print("\n📊 RAGAS EVALUATION RESULTS:")
            print("=" * 50)
            for metric, score in result.items():
                print(f"{metric:20}: {score:.4f}")
            
            # Save results
            results_data = {
                "evaluation_framework": "ragas",
                "overall_scores": dict(result),
                "questions_evaluated": len(questions),
                "test_questions": test_questions,
                "individual_results": []
            }
            
            # Add individual question results
            for i, (q, a, c, gt) in enumerate(zip(questions, answers, contexts, ground_truths)):
                results_data["individual_results"].append({
                    "question": q,
                    "answer": a,
                    "contexts": c,
                    "ground_truth": gt
                })
            
            # Save to file
            with open("working_ragas_results.json", "w") as f:
                json.dump(results_data, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to 'working_ragas_results.json'")
            
            # Generate analysis
            print("\n📄 PERFORMANCE ANALYSIS:")
            print("=" * 50)
            
            avg_score = sum(result.values()) / len(result)
            print(f"Average Score: {avg_score:.4f}")
            
            # Metric-specific analysis
            for metric, score in result.items():
                if score >= 0.8:
                    status = "🎉 Excellent"
                elif score >= 0.7:
                    status = "✅ Good"
                elif score >= 0.6:
                    status = "⚠️  Fair"
                else:
                    status = "❌ Poor"
                
                print(f"{metric:20}: {score:.4f} {status}")
            
            print(f"\n🎯 OVERALL PERFORMANCE:")
            if avg_score >= 0.8:
                print("🎉 Excellent! Your QTL RAG system is performing very well.")
            elif avg_score >= 0.7:
                print("✅ Good performance! Minor optimizations could improve results.")
            elif avg_score >= 0.6:
                print("⚠️  Fair performance. Consider improving retrieval or generation.")
            else:
                print("❌ Poor performance. Significant improvements needed.")
            
            # Specific recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            
            if result.get('context_precision', 0) < 0.7:
                print("📝 Improve context precision: Review chunking strategy and retrieval parameters")
            
            if result.get('context_recall', 0) < 0.7:
                print("🔍 Improve context recall: Consider increasing retrieval count or improving embeddings")
            
            if result.get('faithfulness', 0) < 0.8:
                print("🎯 Improve faithfulness: Enhance prompts to better ground answers in context")
            
            if result.get('answer_relevancy', 0) < 0.7:
                print("💬 Improve answer relevancy: Optimize answer generation prompts")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            return None

def main():
    """Run the working evaluation."""
    try:
        evaluator = WorkingQTLEvaluator()
        results = evaluator.run_evaluation()
        
        if results:
            print("\n✅ Working Ragas evaluation completed successfully!")
            print("\n🔗 Next steps:")
            print("1. Review detailed results in 'working_ragas_results.json'")
            print("2. Implement recommended improvements")
            print("3. Re-run evaluation to measure progress")
        else:
            print("\n❌ Evaluation failed.")
            
    except Exception as e:
        print(f"❌ Error running evaluation: {str(e)}")

if __name__ == "__main__":
    main() 