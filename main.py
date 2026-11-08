import os
from dotenv import load_dotenv
from deepeval import evaluate 
from dataset.test_cases import test_cases
from metrics.tutor_dag import tutor_evaluation_metric
load_dotenv()
if __name__ == "__main__":
    results=evaluate(
        test_cases=test_cases,
        metrics=[tutor_evaluation_metric],
    )
    print("Evaluation completed successfully")