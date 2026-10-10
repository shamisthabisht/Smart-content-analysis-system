import unittest
import pandas as pd
from sklearn.model_selection import train_test_split

class TestSplitting(unittest.TestCase):
    def test_stratified_split_integrity(self):
        # Create dummy balanced dataset
        df = pd.DataFrame({
            "text": [f"sample text {i}" for i in range(100)],
            "label": [0] * 50 + [1] * 50
        })

        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
        
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)
        self.assertEqual(train_df["label"].value_counts()[0], 40)
        self.assertEqual(train_df["label"].value_counts()[1], 40)
        self.assertEqual(test_df["label"].value_counts()[0], 10)
        self.assertEqual(test_df["label"].value_counts()[1], 10)

        # Ensure no overlap
        train_texts = set(train_df["text"])
        test_texts = set(test_df["text"])
        self.assertEqual(len(train_texts.intersection(test_texts)), 0)

if __name__ == "__main__":
    unittest.main()
