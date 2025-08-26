# import ai_trainer
# import torch 
import ai_agent

if __name__=="__main__":
    #if not lora-adapter file:
        # trainer = ai_trainer.trainer.train()
    answer, ctx_docs = ai_agent.rag_answer()
    print("ANSWER:\n", answer)
    print("\n--- Retrieved Chunks ---")
    for i, d in enumerate(ctx_docs, 1):
        print(f"[{i}] {d.page_content[:200]}...")