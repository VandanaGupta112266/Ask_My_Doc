import json
import sys
import os

# Define your production quality thresholds
# If a score is below these numbers, the build will "FAIL"
THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.80,
    "context_recall": 0.70,
    "context_precision": 0.70
}

def check_quality(report_path="data/eval/baseline_report.json"):
    print(f"\n🔍 Checking Quality Report: {report_path}")
    
    if not os.path.exists(report_path):
        print(f"❌ Error: Report file {report_path} not found. Please run the evaluation pipeline first.")
        sys.exit(1)
        
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        sys.exit(1)
        
    averages = report.get("global_averages", {})
    failed = False
    
    print("\n" + "="*60)
    print(f"{'METRIC':20} | {'SCORE':8} | {'TARGET':8} | {'STATUS'}")
    print("-" * 60)
    
    for metric, threshold in THRESHOLDS.items():
        score = averages.get(metric, 0)
        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        print(f"{metric:20} | {score:8.4f} | {threshold:8.2f} | {status}")
        
        if score < threshold:
            failed = True
            
    print("="*60)
            
    if failed:
        print("\n❌ QUALITY GATE FAILED: RAG performance is below production standards.")
        print("Please review the detailed results and optimize your prompt or retrieval settings.")
        sys.exit(1)
    else:
        print("\n✅ QUALITY GATE PASSED: System meets production quality standards.")
        sys.exit(0)

if __name__ == "__main__":
    check_quality()
