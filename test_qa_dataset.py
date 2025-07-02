import json
from rag_chatbot import QTLChatbot
from typing import List, Dict, Any
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score
import os

class QADatasetTester:
    def __init__(self, dataset_path: str = "qtl_qa_dataset.json", wait_time: float = 1.0):
        """Initialize the QA dataset tester.
        
        Args:
            dataset_path: Path to the QA dataset file
            wait_time: Time to wait between API calls in seconds
        """
        self.chatbot = QTLChatbot()
        self.dataset_path = dataset_path
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.wait_time = wait_time
        
        # Create results directory structure
        self.results_dir = "results"
        self.log_dir = os.path.join(self.results_dir, "logs")
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        
        for directory in [self.results_dir, self.log_dir, self.metrics_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize log file
        self.log_file = os.path.join(self.log_dir, f"test_log_{self.timestamp}.txt")
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"QTL Chatbot Test Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    
    def calculate_metrics(self, pred: str, gold: str) -> Dict[str, float]:
        """Calculate evaluation metrics between predicted and gold answers."""
        # Exact match
        em = int(pred.strip().lower() == gold.strip().lower())
        
        # F1 score calculation
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        common = pred_tokens & gold_tokens
        
        if len(common) > 0:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
            
        return {
            "exact_match": em,
            "f1_score": f1,
            "precision": precision if len(common) > 0 else 0.0,
            "recall": recall if len(common) > 0 else 0.0
        }
    
    def test_question(self, qa_pair: Dict[str, str]) -> Dict[str, Any]:
        """Test a single question against the chatbot."""
        try:
            # Get the chatbot's answer
            start_time = time.time()
            chatbot_answer = self.chatbot.process_query(qa_pair['question'])
            response_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self.calculate_metrics(chatbot_answer, qa_pair['answer'])
            
            # Wait to avoid rate limits
            time.sleep(self.wait_time)
            
            return {
                "question": qa_pair['question'],
                "expected_answer": qa_pair['answer'],
                "chatbot_answer": chatbot_answer,
                "ground_truth_context": qa_pair['ground_truth_context'],
                "response_time": response_time,
                "metrics": metrics
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
                self.log(f"Precision: {result['metrics']['precision']:.3f}")
                self.log(f"Recall: {result['metrics']['recall']:.3f}")
                self.log(f"F1: {result['metrics']['f1_score']:.3f}")
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
        # Save detailed results
        results_file = os.path.join(self.metrics_dir, f"detailed_results_{self.timestamp}.json")
        
        # Calculate aggregate metrics
        exact_matches = [r['metrics']['exact_match'] for r in self.results]
        f1_scores = [r['metrics']['f1_score'] for r in self.results]
        precisions = [r['metrics']['precision'] for r in self.results]
        recalls = [r['metrics']['recall'] for r in self.results]
        
        results_data = {
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(self.results),
            "average_response_time": avg_time,
            "wait_time_between_calls": self.wait_time,
            "aggregate_metrics": {
                "exact_match_rate": np.mean(exact_matches),
                "f1_score": np.mean(f1_scores),
                "precision": np.mean(precisions),
                "recall": np.mean(recalls)
            },
            "results": self.results
        }
        
        try:
            # Save detailed results
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2)
            self.log(f"\nDetailed results saved to {results_file}")
            
            # Save summary metrics
            summary_file = os.path.join(self.metrics_dir, f"summary_{self.timestamp}.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("QTL Chatbot Test Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Questions: {len(self.results)}\n")
                f.write(f"Average Response Time: {avg_time:.2f} seconds\n")
                f.write(f"Wait Time Between Calls: {self.wait_time} seconds\n\n")
                f.write("Evaluation Metrics:\n")
                f.write(f"Exact Match Rate: {np.mean(exact_matches)*100:.1f}%\n")
                f.write(f"F1 Score: {np.mean(f1_scores):.3f}\n")
                f.write(f"Precision: {np.mean(precisions):.3f}\n")
                f.write(f"Recall: {np.mean(recalls):.3f}\n")
            self.log(f"Summary saved to {summary_file}")
            
        except Exception as e:
            self.log(f"Error saving results: {str(e)}")
    
    def print_summary(self) -> None:
        """Write a summary of the test results to the log file."""
        if not self.results:
            self.log("No test results available!")
            return
        
        # Calculate aggregate metrics
        exact_matches = [r['metrics']['exact_match'] for r in self.results]
        f1_scores = [r['metrics']['f1_score'] for r in self.results]
        precisions = [r['metrics']['precision'] for r in self.results]
        recalls = [r['metrics']['recall'] for r in self.results]
        
        self.log("\nTest Results Summary:")
        self.log("=" * 50)
        self.log(f"Total Questions Tested: {len(self.results)}")
        self.log(f"Average Response Time: {sum(r['response_time'] for r in self.results) / len(self.results):.2f} seconds")
        self.log(f"Wait Time Between Calls: {self.wait_time} seconds")
        self.log("\nEvaluation Metrics:")
        self.log(f"Exact Match Rate: {np.mean(exact_matches)*100:.1f}%")
        self.log(f"F1 Score: {np.mean(f1_scores):.3f}")
        self.log(f"Precision: {np.mean(precisions):.3f}")
        self.log(f"Recall: {np.mean(recalls):.3f}")
        self.log("=" * 50)

def main():
    # Initialize the tester with a 4-second wait time between calls (15 requests per minute)
    tester = QADatasetTester(wait_time=4.0)
    
    # Run the tests
    tester.run_tests()
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    main() 