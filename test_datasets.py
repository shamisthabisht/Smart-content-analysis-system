import json
from datasets import load_dataset

print("Testing Dataset Discovery...")

# 1. AI vs Human text candidate: Hello-SimpleAI/HC3
try:
    print("Testing HC3 dataset...")
    hc3 = load_dataset("Hello-SimpleAI/hc3", "all", split="train[:500]")
    print(f"HC3 sample keys: {hc3[0].keys()}")
    print(f"HC3 sample question: {hc3[0]['question'][:100]}")
    print(f"HC3 human answers count: {len(hc3[0]['human_answers'])}, chatgpt answers: {len(hc3[0]['chatgpt_answers'])}")
    print("HC3 loaded successfully!")
except Exception as e:
    print(f"HC3 error: {e}")

# 1b. AI vs Human text candidate: aadityaubhat/GPT-wiki-intro
try:
    print("\nTesting GPT-wiki-intro dataset...")
    wiki_intro = load_dataset("aadityaubhat/GPT-wiki-intro", split="train[:500]")
    print(f"GPT-wiki-intro sample keys: {wiki_intro[0].keys()}")
    print(f"GPT-wiki-intro sample title: {wiki_intro[0]['title']}")
    print(f"GPT-wiki-intro sample human: {wiki_intro[0]['wiki_intro'][:100]}")
    print(f"GPT-wiki-intro sample generated: {wiki_intro[0]['generated_intro'][:100]}")
    print("GPT-wiki-intro loaded successfully!")
except Exception as e:
    print(f"GPT-wiki-intro error: {e}")

# 2. Grammar Correction candidate: jfleg
try:
    print("\nTesting JFLEG dataset...")
    jfleg = load_dataset("jhu-clsp/jfleg", split="test[:100]")
    print(f"JFLEG sample keys: {jfleg[0].keys()}")
    print(f"JFLEG sample sentence: {jfleg[0]['sentence']}")
    print(f"JFLEG sample corrections: {jfleg[0]['corrections']}")
    print("JFLEG loaded successfully!")
except Exception as e:
    print(f"JFLEG error: {e}")

# 2b. Grammar Correction candidate: c4_200m GEC or coedit or similar
try:
    print("\nTesting Grammarly CoEdit dataset...")
    coedit = load_dataset("grammarly/coedit", split="train[:100]")
    print(f"CoEdit sample keys: {coedit[0].keys()}")
    print(f"CoEdit sample task: {coedit[0]['task']}, src: {coedit[0]['src']}, tgt: {coedit[0]['tgt']}")
    print("CoEdit loaded successfully!")
except Exception as e:
    print(f"CoEdit error: {e}")
