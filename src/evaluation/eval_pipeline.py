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
from langfuse.decorators import observe, langfuse_context
from src.retrieval.vector_store import get_vector_store, get_embeddings
from dotenv import load_dotenv

load_dotenv()

@observe()
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
    trace_ids = []
    
    vector_store = get_vector_store()
    
    # 3. Collect RAG answers for each question
    for entry in golden_data[:1]:
        question = entry["question"]
        q_type = entry.get("type", "standard")
        print(f"\nEvaluating [{q_type}]: {question}")
        
        # Tag the trace in Langfuse and get its ID
        langfuse_context.update_current_trace(
            tags=["evaluation"],
            metadata={"question_type": q_type}
        )
        current_trace_id = langfuse_context.get_current_trace_id()
        trace_ids.append(current_trace_id)
        
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
    
    # Pro-Tip: For free-tier LLMs, we MUST run with small batch sizes 
    # and longer timeouts to avoid 429s and TimeoutErrors.
    # Run evaluation (removing incompatible is_async parameter)
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=eval_llm,
        embeddings=get_embeddings()
    )
    
    # 6. Push Scores to Langfuse Dashboard
    df = result.to_pandas()
    print("\n--- Syncing Scores to Langfuse Dashboard ---")
    
    for i, trace_id in enumerate(trace_ids):
        for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            if metric in df.columns:
                score_val = df.iloc[i][metric]
                if not pd.isna(score_val):
                    # Use the underlying client instance for specific trace scoring
                    langfuse_context.client_instance.score(
                        name=metric,
                        value=float(score_val),
                        trace_id=trace_id
                    )
    
    # 6. Report Results
    print("\n--- Evaluation Summary ---")
    
    # Safe column access: only print columns that actually exist in the result
    cols_to_show = ['question', 'question_type', 'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
    available_cols = [c for c in cols_to_show if c in df.columns]
    
    if available_cols:
        print(df[available_cols].to_string(index=False))
    else:
        print("Detailed results:")
        print(df.head())
    
    # Calculate averages from the dataframe (Most robust way)
    print("\n--- Global Averages ---")
    numeric_df = df.select_dtypes(include=['number'])
    averages = numeric_df.mean().to_dict()
    
    for metric, score in averages.items():
        print(f"{metric}: {score:.4f}")
    
    # Save to CSV for tracking
    report_path_csv = "data/eval/baseline_report.csv"
    report_path_json = "data/eval/baseline_report.json"
    os.makedirs("data/eval", exist_ok=True)
    
    # Save CSV
    df.to_csv(report_path_csv, index=False)
    
    # Save JSON (more structured)
    import datetime
    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "global_averages": averages,
        "detailed_results": df.to_dict(orient="records")
    }
    with open(report_path_json, "w") as f:
        json.dump(report_data, f, indent=4)
        
    print(f"\nFull report saved to:")
    print(f"- CSV: {report_path_csv}")
    print(f"- JSON: {report_path_json}")

if __name__ == "__main__":
    run_evaluation()
