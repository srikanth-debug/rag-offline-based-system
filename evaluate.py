import requests
import json
import pandas as pd

API_URL = "http://127.0.0.1:8000/ask"
EVAL_QUESTIONS_PATH = "eval/questions.jsonl"

def main():
    try:
        with open(EVAL_QUESTIONS_PATH, 'r') as f:
            eval_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Evaluation file not found at {EVAL_QUESTIONS_PATH}")
        return

    results = []
    print(f"Running evaluation against {len(eval_data)} questions...")
    for item in eval_data:
        question = item['question']
        
        try:
            payload = {"query": question}
            response = requests.post(API_URL, json=payload, timeout=900)
            response.raise_for_status()
            data = response.json()
            
            # Check if the system correctly abstained
            is_correct_abstention = (item['gold_answer'] == "ABSTAIN" and "cannot answer" in data['answer'])
            
            results.append({
                "question": question,
                "generated_answer": data['answer'],
                "gold_answer": item['gold_answer'],
                "confidence": data['confidence'],
                "correct_abstention": is_correct_abstention
            })
        except requests.exceptions.RequestException as e:
            print(f"Failed to process question: '{question}'. Error: {e}")

    # Display results
    df = pd.DataFrame(results)
    print("\n--- Evaluation Results ---")
    print(df[['question', 'confidence', 'correct_abstention']])

    # Calculate overall metrics
    abstain_questions = df[df['gold_answer'] == 'ABSTAIN']
    if not abstain_questions.empty:
        abstention_precision = abstain_questions['correct_abstention'].sum() / len(abstain_questions)
        print(f"\nAbstention Precision: {abstention_precision:.2f}")
    else:
        print("\nNo abstention questions in eval set.")

if __name__ == "__main__":
    main()