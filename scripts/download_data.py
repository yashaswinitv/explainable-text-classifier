from datasets import load_dataset

if __name__ == "__main__":
    load_dataset("ag_news")  # caches locally in ~/.cache/huggingface
    print("Downloaded AG News.")
