import json
from rag_chatbot import QTLChatbot
from typing import List, Dict, Any
import time
from datetime import datetime
import os
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

class DeepEvalTester:
    def __init__(self, dataset_path: str = "qtl_qa_dataset.json", wait_time: float = 4.0):
        """Initialize the DeepEval tester.
        
        Args:
            dataset_path: Path to the QA dataset file
            wait_time: Time to wait between API calls in seconds (15 requests per minute)
        """
        self.chatbot = QTLChatbot()
        self.dataset_path = dataset_path
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.wait_time = wait_time
        
        # Create results directory structure
        self.results_dir = "deepeval_results"
        self.log_dir = os.path.join(self.results_dir, "logs")
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        
        for directory in [self.results_dir, self.log_dir, self.metrics_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize log file
        self.log_file = os.path.join(self.log_dir, f"deepeval_test_log_{self.timestamp}.txt")
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"DeepEval Test Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log(self, message: str) -> None:
        """Write a message to the log file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
    
    def load_dataset(self) -> List[Dict[str, str]]:
        """Load the QA dataset from file."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['dataset']
        except Exception as e:
            self.log(f"Error loading dataset: {str(e)}")
            return []
    
    def create_test_case(self, qa_pair: Dict[str, str], chatbot_answer: str) -> LLMTestCase:
        """Create a DeepEval test case from a QA pair."""
        # Convert ground truth context to a list of strings if it's not already
        context = qa_pair['ground_truth_context']
        if isinstance(context, str):
            # Split by newlines and filter out empty lines
            context = [line.strip() for line in context.split('\n') if line.strip()]
        
        return LLMTestCase(
            input=qa_pair['question'],
            actual_output=chatbot_answer,
            expected_output=qa_pair['answer'],
            retrieval_context=context
        )
    
    def evaluate_response(self, test_case: LLMTestCase) -> Dict[str, float]:
        """Evaluate the response using our RAG chatbot."""
        try:
            # Get the chatbot's evaluation of the response
            evaluation_prompt = f"""
            Question: {test_case.input}
            Expected Answer: {test_case.expected_output}
            Actual Answer: {test_case.actual_output}
            
            Please evaluate the actual answer on a scale of 0 to 1 for:
            1. Accuracy: How well does it match the expected answer?
            2. Completeness: Does it include all necessary information?
            3. Relevance: Is it directly answering the question?
            
            Format your response as a JSON with these three scores.
            """
            
            evaluation_response = self.chatbot.process_query(evaluation_prompt)
            
            # Parse the evaluation scores
            try:
                # Try to parse as JSON
                scores = json.loads(evaluation_response)
            except:
                # If not JSON, try to extract scores from text
                scores = {
                    "accuracy": 0.0,
                    "completeness": 0.0,
                    "relevance": 0.0
                }
                # Look for scores in the text
                for line in evaluation_response.split('\n'):
                    if 'accuracy' in line.lower():
                        try:
                            scores['accuracy'] = float(line.split(':')[-1].strip())
                        except:
                            pass
                    elif 'completeness' in line.lower():
                        try:
                            scores['completeness'] = float(line.split(':')[-1].strip())
                        except:
                            pass
                    elif 'relevance' in line.lower():
                        try:
                            scores['relevance'] = float(line.split(':')[-1].strip())
                        except:
                            pass
            
            return scores
            
        except Exception as e:
            self.log(f"Error evaluating response: {str(e)}")
            return {
                "accuracy": 0.0,
                "completeness": 0.0,
                "relevance": 0.0
            }
    
    def test_question(self, qa_pair: Dict[str, str]) -> Dict[str, Any]:
        """Test a single question against the chatbot."""
        try:
            # Get the chatbot's answer
            start_time = time.time()
            chatbot_answer = self.chatbot.process_query(qa_pair['question'])
            response_time = time.time() - start_time
            
            # Create test case
            test_case = self.create_test_case(qa_pair, chatbot_answer)
            
            # Evaluate the response
            evaluation_scores = self.evaluate_response(test_case)
            
            # Wait to avoid rate limits
            time.sleep(self.wait_time)
            
            return {
                "question": qa_pair['question'],
                "expected_answer": qa_pair['answer'],
                "chatbot_answer": chatbot_answer,
                "ground_truth_context": qa_pair['ground_truth_context'],
                "response_time": response_time,
                "metrics": evaluation_scores
            }
        except Exception as e:
            self.log(f"Error testing question: {str(e)}")
            return None
    
    def run_tests(self) -> None:
        """Run tests on all questions in the dataset."""
        self.log("Loading dataset...")
        qa_pairs = self.load_dataset()
        
        if not qa_pairs:
            self.log("No questions found in dataset!")
            return
        
        self.log(f"\nTesting {len(qa_pairs)} questions...")
        total_time = 0
        
        for i, qa_pair in enumerate(qa_pairs, 1):
            self.log(f"\nQuestion {i}/{len(qa_pairs)}:")
            self.log("-" * 80)
            self.log(f"Q: {qa_pair['question']}")
            
            result = self.test_question(qa_pair)
            if result:
                self.results.append(result)
                total_time += result['response_time']
                
                self.log(f"Expected: {result['expected_answer']}")
                self.log(f"Chatbot: {result['chatbot_answer']}")
                for metric_name, score in result['metrics'].items():
                    self.log(f"{metric_name}: {score:.3f}")
                self.log("-" * 80)
                
                # Log wait time
                if i < len(qa_pairs):
                    self.log(f"Waiting {self.wait_time} seconds before next question...")
        
        # Calculate statistics
        avg_time = total_time / len(qa_pairs) if qa_pairs else 0
        
        # Save results
        self.save_results(avg_time)
        
        self.log("\nTesting complete!")
        self.log(f"Average response time: {avg_time:.2f} seconds")
    
    def save_results(self, avg_time: float) -> None:
        """Save test results to files."""
        if not self.results:
            self.log("No test results to save!")
            return
            
        # Save detailed results
        results_file = os.path.join(self.metrics_dir, f"deepeval_results_{self.timestamp}.json")
        
        # Calculate aggregate metrics
        metric_averages = {}
        if self.results and self.results[0]['metrics']:
            for metric_name in self.results[0]['metrics'].keys():
                scores = [r['metrics'][metric_name] for r in self.results if 'metrics' in r and metric_name in r['metrics']]
                if scores:  # Only calculate average if we have scores
                    metric_averages[metric_name] = sum(scores) / len(scores)
        
        results_data = {
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(self.results),
            "successful_tests": len([r for r in self.results if r is not None]),
            "average_response_time": avg_time,
            "wait_time_between_calls": self.wait_time,
            "aggregate_metrics": metric_averages,
            "results": self.results
        }
        
        try:
            # Save detailed results
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2)
            self.log(f"\nDetailed results saved to {results_file}")
            
            # Save summary metrics
            summary_file = os.path.join(self.metrics_dir, f"deepeval_summary_{self.timestamp}.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("DeepEval Test Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Questions: {len(self.results)}\n")
                f.write(f"Successful Tests: {len([r for r in self.results if r is not None])}\n")
                f.write(f"Average Response Time: {avg_time:.2f} seconds\n")
                f.write(f"Wait Time Between Calls: {self.wait_time} seconds\n\n")
                f.write("Evaluation Metrics:\n")
                for metric_name, score in metric_averages.items():
                    f.write(f"Average {metric_name}: {score:.3f}\n")
            self.log(f"Summary saved to {summary_file}")
            
        except Exception as e:
            self.log(f"Error saving results: {str(e)}")
    
    def print_summary(self) -> None:
        """Write a summary of the test results to the log file."""
        if not self.results:
            self.log("No test results available!")
            return
        
        # Calculate aggregate metrics
        metric_averages = {}
        if self.results and self.results[0]['metrics']:
            for metric_name in self.results[0]['metrics'].keys():
                scores = [r['metrics'][metric_name] for r in self.results if 'metrics' in r and metric_name in r['metrics']]
                if scores:  # Only calculate average if we have scores
                    metric_averages[metric_name] = sum(scores) / len(scores)
        
        self.log("\nTest Results Summary:")
        self.log("=" * 50)
        self.log(f"Total Questions Tested: {len(self.results)}")
        self.log(f"Successful Tests: {len([r for r in self.results if r is not None])}")
        self.log(f"Average Response Time: {sum(r['response_time'] for r in self.results if r is not None) / len(self.results):.2f} seconds")
        self.log(f"Wait Time Between Calls: {self.wait_time} seconds")
        self.log("\nEvaluation Metrics:")
        for metric_name, score in metric_averages.items():
            self.log(f"Average {metric_name}: {score:.3f}")
        self.log("=" * 50)

def main():
    # Initialize the tester with a 4-second wait time between calls (15 requests per minute)
    tester = DeepEvalTester(wait_time=4.0)
    
    # Run the tests
    tester.run_tests()
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    main() 