import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from src.generation.qa_pipeline import generate_answer, get_llm
from src.retrieval.vector_store import get_vector_store, get_embeddings
from dotenv import load_dotenv

load_dotenv()

def run_evaluation(dataset_path="data/eval/golden_dataset.json"):
    print(f"--- Starting Evaluation on {dataset_path} ---")
    
    # 1. Load Golden Dataset
    with open(dataset_path, "r") as f:
        golden_data = json.load(f)
    
    # 2. Initialize results storage
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    question_types = []
    
    vector_store = get_vector_store()
    
    # 3. Run Pipeline for each question
    for entry in golden_data:
        question = entry["question"]
        q_type = entry.get("type", "standard")
        print(f"\nEvaluating [{q_type}]: {question}")
        
        # Get answer and retrieved chunks
        answer, chunks = generate_answer(question, vector_store)
        
        # Format contexts as a list of strings (required by RAGAS)
        retrieved_contexts = [c.page_content for c in chunks]
        
        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(entry["ground_truth"])
        question_types.append(q_type)
    
    # 4. Prepare dataset for RAGAS
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "question_type": question_types
    }
    dataset = Dataset.from_dict(data)
    
    # 5. Run RAGAS Evaluation
    print("\n--- Calculating RAGAS Scores ---")
    # We use our OpenRouter LLM for evaluation as well
    eval_llm = get_llm()
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=eval_llm,
        embeddings=get_embeddings(),
        raise_exceptions=False
    )
    
    # 6. Report Results
    df = result.to_pandas()
    print("\n--- Evaluation Summary ---")
    
    # Safe column access: only print columns that actually exist in the result
    cols_to_show = ['question', 'question_type', 'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
    available_cols = [c for c in cols_to_show if c in df.columns]
    
    if available_cols:
        print(df[available_cols].to_string(index=False))
    else:
        print("Detailed results:")
        print(df.head())
    
    # Calculate averages
    print("\n--- Global Averages ---")
    for metric, score in result.scores.items():
        print(f"{metric}: {score:.4f}")
    
    # Save to CSV for tracking
    report_path = "data/eval/baseline_report.csv"
    os.makedirs("data/eval", exist_ok=True)
    df.to_csv(report_path, index=False)
    print(f"\nFull report saved to: {report_path}")

if __name__ == "__main__":
    run_evaluation()
