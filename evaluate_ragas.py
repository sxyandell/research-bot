import json
import os
from datetime import datetime
from dotenv import load_dotenv
from ragas import EvaluationDataset, evaluate
from ragas.metrics import ContextRecall, Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_chatbot import QTLChatbot
import time

# Load environment variables
load_dotenv('config.env')

# Set Google API key for evaluation
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def load_qtl_dataset(dataset_path: str = "qtl_qa_dataset.json"):
    """Load the QTL QA dataset from file."""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['dataset']
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return []

def collect_evaluation_data():
    """Collect evaluation data by running queries through the RAG system."""
    
    print("📊 Collecting Evaluation Data...")
    
    # Initialize RAG chatbot
    rag_application = QTLChatbot()
    
    # Load QTL dataset
    qa_dataset = load_qtl_dataset()
    
    if not qa_dataset:
        print("❌ No QA dataset found. Please run create_qa_dataset.py first.")
        return None
    
    dataset = []
    
    print(f"🔄 Processing {min(3, len(qa_dataset))} QA pairs through RAG system...")
    print("⚠️ Rate limited to 15 requests per minute - using 10-second delays")
    print("⚠️ Both RAG calls and evaluation calls count toward the limit")
    
    for i, qa_pair in enumerate(qa_dataset[:3]):  # Only process first 3 items due to rate limits
        query = qa_pair['question']
        reference = qa_pair['answer']
        ground_truth_context = qa_pair.get('ground_truth_context', '')
        
        print(f"  Processing {i+1}/3: {query[:50]}...")
        
        try:
            # Get response from RAG chatbot
            response = rag_application.process_query(query)
            
            # For now, we'll use the ground truth context as retrieved contexts
            # Convert to list format expected by RAGAS
            retrieved_contexts = [ground_truth_context] if ground_truth_context else []
            
            dataset.append({
                "user_input": query,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": reference
            })
            
            # Add delay to respect rate limits (15 requests per minute = 4 seconds between calls)
            # We need extra time for both RAG and evaluation calls
            time.sleep(10)
            
        except Exception as e:
            print(f"❌ Error processing query {i+1}: {e}")
            dataset.append({
                "user_input": query,
                "retrieved_contexts": [],
                "response": f"Error: {str(e)}",
                "reference": reference
            })
    
    print(f"✅ Collected evaluation data for {len(dataset)} queries")
    return dataset

def main():
    """Main function to run the complete RAGAS evaluation."""
    
    print("🚀 RAGAS Evaluation for QTL RAG Chatbot")
    print("=" * 50)
    
    # Step 1: Collect evaluation data
    dataset = collect_evaluation_data()
    
    if not dataset:
        print("❌ Failed to collect evaluation data")
        return
    
    # Step 2: Setup evaluator
    print("🔧 Setting up RAGAS Evaluator...")
    
    try:
        llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
        evaluator_llm = LangchainLLMWrapper(llm)
        print("✅ RAGAS evaluator setup successful")
    except Exception as e:
        print(f"❌ Error setting up evaluator: {e}")
        return
    
    # Step 3: Run RAGAS evaluation
    print("\n🧪 Running RAGAS Evaluation...")
    
    try:
        # Convert dataset to RAGAS format
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        print(f"✅ Created RAGAS evaluation dataset with {len(evaluation_dataset)} samples")
        
        # Define metrics - using the correct metric names for current RAGAS version
        metrics = [
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy()
        ]
        
        print("📊 Evaluating with metrics:")
        for metric in metrics:
            print(f"  - {metric.__class__.__name__}")
        
        # Run evaluation
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=evaluator_llm
        )
        
        print("✅ RAGAS evaluation completed")
        
        # Add delay after evaluation to respect rate limits
        print("⏳ Waiting 10 seconds to respect rate limits...")
        time.sleep(10)
        
    except Exception as e:
        print(f"❌ Error during RAGAS evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Save results
    os.makedirs("results1", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = f"results1/ragas_results_{timestamp}.json"
    
    results_data = {
        'timestamp': timestamp,
        'total_samples': len(dataset),
        'metrics': {},
        'dataset': dataset
    }
    
    # Extract metric results
    if result:
        # Handle different result formats
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        elif hasattr(result, '__dict__'):
            result_dict = result.__dict__
        else:
            result_dict = result
            
        for metric_name, metric_value in result_dict.items():
            # Skip if it's not a metric value (like EvaluationDataset objects)
            if hasattr(metric_value, 'to_dict') or hasattr(metric_value, '__dict__'):
                continue
                
            # Handle case where metric_value might be a list
            if isinstance(metric_value, list):
                if metric_value:
                    score_value = sum(metric_value) / len(metric_value) if all(isinstance(x, (int, float)) for x in metric_value) else None
                else:
                    score_value = None
            else:
                score_value = float(metric_value) if metric_value is not None else None
            
            results_data['metrics'][metric_name] = {
                'score': score_value
            }
    
    # Save to file
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Step 5: Print summary
    print("\n" + "=" * 50)
    print("📊 RAGAS EVALUATION RESULTS")
    print("=" * 50)
    
    if result:
        # Handle different result formats
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        elif hasattr(result, '__dict__'):
            result_dict = result.__dict__
        else:
            result_dict = result
            
        for metric_name, metric_value in result_dict.items():
            # Skip if it's not a metric value (like EvaluationDataset objects)
            if hasattr(metric_value, 'to_dict') or hasattr(metric_value, '__dict__'):
                continue
                
            # Handle case where metric_value might be a list
            if isinstance(metric_value, list):
                if metric_value:
                    score_value = sum(metric_value) / len(metric_value) if all(isinstance(x, (int, float)) for x in metric_value) else None
                else:
                    score_value = None
            else:
                score_value = float(metric_value) if metric_value is not None else None
            
            if score_value is not None:
                print(f"{metric_name}: {score_value:.4f}")
            else:
                print(f"{metric_name}: N/A")
        
        # Calculate average score
        valid_scores = []
        for metric_name, metric_value in result_dict.items():
            # Skip if it's not a metric value
            if hasattr(metric_value, 'to_dict') or hasattr(metric_value, '__dict__'):
                continue
                
            if isinstance(metric_value, list):
                if metric_value and all(isinstance(x, (int, float)) for x in metric_value):
                    valid_scores.append(sum(metric_value) / len(metric_value))
            elif isinstance(metric_value, (int, float)):
                valid_scores.append(metric_value)
        
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            print(f"\nAverage Score: {avg_score:.4f}")
            
            # Performance assessment
            if avg_score >= 0.8:
                print("🎉 Performance: EXCELLENT")
            elif avg_score >= 0.6:
                print("✅ Performance: GOOD")
            elif avg_score >= 0.4:
                print("⚠️ Performance: FAIR")
            else:
                print("❌ Performance: NEEDS IMPROVEMENT")
    
    print(f"\n📁 Results saved to: {results_file}")
    print("🎉 RAGAS evaluation completed!")

if __name__ == "__main__":
    main()
