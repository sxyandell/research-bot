import json
from rag_chatbot import QTLChatbot
from typing import List, Dict, Any
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score
import os

class QADatasetTester:
    def __init__(self, dataset_path: str = "qtl_qa_dataset.json"):
        """Initialize the QA dataset tester."""
        self.chatbot = QTLChatbot()
        self.dataset_path = dataset_path
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = "test_logs"
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
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
            self.log(f"\nTesting question {i}/{len(qa_pairs)}")
            self.log(f"Question: {qa_pair['question']}")
            
            result = self.test_question(qa_pair)
            if result:
                self.results.append(result)
                total_time += result['response_time']
                
                self.log(f"Expected answer: {result['expected_answer']}")
                self.log(f"Chatbot answer: {result['chatbot_answer']}")
                self.log(f"Response time: {result['response_time']:.2f} seconds")
                self.log(f"Exact Match: {result['metrics']['exact_match']}")
                self.log(f"F1 Score: {result['metrics']['f1_score']:.3f}")
        
        # Calculate statistics
        avg_time = total_time / len(qa_pairs) if qa_pairs else 0
        
        # Save results
        self.save_results(avg_time)
        
        self.log("\nTesting complete!")
        self.log(f"Average response time: {avg_time:.2f} seconds")
    
    def save_results(self, avg_time: float) -> None:
        """Save test results to a file."""
        filename = os.path.join(self.log_dir, f"qa_test_results_{self.timestamp}.json")
        
        # Calculate aggregate metrics
        exact_matches = [r['metrics']['exact_match'] for r in self.results]
        f1_scores = [r['metrics']['f1_score'] for r in self.results]
        precisions = [r['metrics']['precision'] for r in self.results]
        recalls = [r['metrics']['recall'] for r in self.results]
        
        results_data = {
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(self.results),
            "average_response_time": avg_time,
            "aggregate_metrics": {
                "exact_match_rate": np.mean(exact_matches),
                "f1_score": np.mean(f1_scores),
                "precision": np.mean(precisions),
                "recall": np.mean(recalls)
            },
            "results": self.results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2)
            self.log(f"\nResults saved to {filename}")
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
        self.log("\nEvaluation Metrics:")
        self.log(f"Exact Match Rate: {np.mean(exact_matches)*100:.1f}%")
        self.log(f"F1 Score: {np.mean(f1_scores):.3f}")
        self.log(f"Precision: {np.mean(precisions):.3f}")
        self.log(f"Recall: {np.mean(recalls):.3f}")
        self.log("=" * 50)

def main():
    # Initialize the tester
    tester = QADatasetTester()
    
    # Run the tests
    tester.run_tests()
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    main() 