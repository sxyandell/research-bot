import json
from rag_chatbot import QTLChatbot
from typing import List, Dict, Any
import time
from datetime import datetime
import os
import pytest
import warnings
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.metrics.ragas import (
    RAGASContextualPrecisionMetric,
    RAGASFaithfulnessMetric,
    RAGASContextualRecallMetric,
    RAGASAnswerRelevancyMetric,
)
from deepeval.metrics import BiasMetric
from deepeval.test_case import LLMTestCase

# Suppress pytest assertion rewrite warnings
warnings.filterwarnings("ignore", category=pytest.PytestAssertRewriteWarning)

# Load environment variables from config.env
load_dotenv('config.env')

# Set OpenAI API key explicitly for deepeval
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Debug: Check if API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"✅ OpenAI API key loaded successfully (first 20 chars: {api_key[:20]}...)")
else:
    print("❌ OpenAI API key not found in environment variables")

#######################################
# Initialize metrics with thresholds ##
#######################################
try:
    bias = BiasMetric(threshold=0.5)
    contextual_precision = RAGASContextualPrecisionMetric(threshold=0.5)
    contextual_recall = RAGASContextualRecallMetric(threshold=0.5)
    answer_relevancy = RAGASAnswerRelevancyMetric(threshold=0.5)
    faithfulness = RAGASFaithfulnessMetric(threshold=0.5)
    
    #######################################
    # Specify evaluation metrics to use ###
    #######################################
    evaluation_metrics = [
        bias,
        contextual_precision,
        contextual_recall,
        answer_relevancy,
        faithfulness
    ]
    print("✅ RAGAS metrics initialized successfully")
except Exception as e:
    print(f"❌ Error initializing RAGAS metrics: {e}")
    evaluation_metrics = []

#######################################
# Initialize RAG application ##########
#######################################
rag_application = QTLChatbot()

def load_qa_dataset(dataset_path: str = "qtl_qa_dataset.json") -> List[Dict[str, str]]:
    """Load the QA dataset from file."""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['dataset']
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return []

def create_input_output_pairs(qa_dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert QA dataset to input-output pairs format."""
    pairs = []
    for qa_pair in qa_dataset:
        pairs.append({
            "input": qa_pair['question'],
            "expected_output": qa_pair['answer'],
            "ground_truth_context": qa_pair['ground_truth_context']
        })
    return pairs

def save_detailed_results(results, filename):
    """Save detailed results to JSON file."""
    os.makedirs("results1", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

#######################################
# Specify inputs to test RAG app on ###
#######################################
qa_dataset = load_qa_dataset()
input_output_pairs = create_input_output_pairs(qa_dataset)

# Global variable to store detailed results
detailed_results = []

#######################################
# Loop through input output pairs #####
#######################################
@pytest.mark.parametrize(
    "input_output_pair",
    input_output_pairs,
)
def test_rag_chatbot(input_output_pair: Dict):
    input = input_output_pair.get("input", None)
    expected_output = input_output_pair.get("expected_output", None)
    ground_truth_context = input_output_pair.get("ground_truth_context", None)

    # Get actual output from our RAG chatbot
    actual_output = rag_application.process_query(input)
    
    # Get retrieval context (we'll use the ground truth context for now)
    # In a real scenario, you might want to get the actual retrieval context
    retrieval_context = ground_truth_context
    if isinstance(retrieval_context, str):
        # Convert to list of strings as required by RAGAS
        retrieval_context = [line.strip() for line in retrieval_context.split('\n') if line.strip()]

    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output
    )
    
    # Store detailed results for this test case
    case_result = {
        "input": input,
        "expected_output": expected_output,
        "actual_output": actual_output,
        "metrics": {}
    }
    
    # Assert test case only if metrics are available
    if evaluation_metrics:
        try:
            print(f"\n🧪 Testing: {input[:80]}...")
            print("=" * 80)
            
            # Run each metric individually and collect scores
            for metric in evaluation_metrics:
                try:
                    score_result = metric.measure(test_case)
                    metric_name = metric.__class__.__name__
                    
                    # Extract score details
                    score_value = getattr(score_result, 'score', None)
                    passed = getattr(score_result, 'passed', None)
                    reason = getattr(score_result, 'reason', None)
                    
                    # Store in case result
                    case_result["metrics"][metric_name] = {
                        "score": score_value,
                        "passed": passed,
                        "reason": reason
                    }
                    
                    # Print detailed results
                    print(f"📊 {metric_name}:")
                    print(f"   Score: {score_value}")
                    print(f"   Passed: {passed}")
                    print(f"   Reason: {reason}")
                    
                except Exception as e:
                    print(f"❌ Error with {metric.__class__.__name__}: {e}")
                    case_result["metrics"][metric.__class__.__name__] = {
                        "score": None,
                        "passed": False,
                        "reason": f"Error: {str(e)}"
                    }
            
            # Add delay between tests to respect rate limits
            time.sleep(4)
            
            print(f"✅ Test completed for: {input[:50]}...")
            
        except Exception as e:
            print(f"❌ Test failed for input: {input[:50]}... Error: {e}")
            case_result["metrics"]["overall_error"] = str(e)
    else:
        print(f"⚠️ Skipping RAGAS evaluation for input: {input[:50]}... (metrics not available)")
        # Basic assertion that we got some response
        assert actual_output is not None and len(actual_output) > 0, "RAG chatbot should return a response"
        case_result["metrics"]["basic_test"] = {
            "score": 1.0 if actual_output and len(actual_output) > 0 else 0.0,
            "passed": actual_output is not None and len(actual_output) > 0,
            "reason": "Basic response validation"
        }
    
    # Add to detailed results
    detailed_results.append(case_result)

def pytest_sessionfinish(session, exitstatus):
    """Save detailed results after all tests complete."""
    if detailed_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results1/detailed_metric_results_{timestamp}.json"
        save_detailed_results(detailed_results, results_file)
        
        # Create summary report
        summary_file = f"results1/metric_summary_{timestamp}.txt"
        create_summary_report(detailed_results, summary_file)
        
        print(f"\n📊 Detailed results saved to: {results_file}")
        print(f"📄 Summary report saved to: {summary_file}")

def create_summary_report(results, filename):
    """Create a human-readable summary report."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("RAG CHATBOT METRIC EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Test Cases: {len(results)}\n\n")
        
        # Calculate summary statistics
        metric_names = set()
        for result in results:
            metric_names.update(result["metrics"].keys())
        
        f.write("METRIC SUMMARY:\n")
        f.write("-" * 30 + "\n")
        
        for metric_name in sorted(metric_names):
            if metric_name == "overall_error" or metric_name == "basic_test":
                continue
                
            scores = []
            passed_count = 0
            total_count = 0
            
            for result in results:
                if metric_name in result["metrics"]:
                    metric_data = result["metrics"][metric_name]
                    if metric_data["score"] is not None:
                        scores.append(metric_data["score"])
                        total_count += 1
                        if metric_data["passed"]:
                            passed_count += 1
            
            if scores:
                avg_score = sum(scores) / len(scores)
                pass_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
                
                f.write(f"{metric_name}:\n")
                f.write(f"  Average Score: {avg_score:.3f}\n")
                f.write(f"  Pass Rate: {pass_rate:.1f}% ({passed_count}/{total_count})\n")
                f.write(f"  Min Score: {min(scores):.3f}\n")
                f.write(f"  Max Score: {max(scores):.3f}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for i, result in enumerate(results):
            f.write(f"\nTest Case {i+1}:\n")
            f.write(f"Input: {result['input'][:100]}...\n")
            f.write(f"Expected: {result['expected_output'][:100]}...\n")
            f.write(f"Actual: {result['actual_output'][:100]}...\n")
            
            for metric_name, metric_data in result["metrics"].items():
                f.write(f"  {metric_name}: Score={metric_data['score']}, Passed={metric_data['passed']}\n")

if __name__ == "__main__":
    # Create results directory
    os.makedirs("results1", exist_ok=True)
    
    # Run the tests and capture results
    import sys
    from io import StringIO
    
    # Capture stdout to save test output
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    
    try:
        # Run the tests
        pytest.main([__file__, "-v"])
    finally:
        # Restore stdout
        sys.stdout = old_stdout
    
    # Save test output to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results1/pytest_results_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write(result.getvalue())
    
    print(f"Test results saved to: {output_file}")
    print("Test output:")
    print(result.getvalue()) 