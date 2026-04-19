import unittest
import pandas as pd

class TestVibeMatch(unittest.TestCase):
    def test_data_loading(self):
        """Test if the processed dataset loads correctly and has the right columns."""
        df = pd.read_csv("processed_spotify_data.csv")
        self.assertTrue('valence' in df.columns)
        self.assertTrue(len(df) > 0)

    def test_value_bounds(self):
        """Test if the audio features are properly Min-Max scaled between 0 and 1."""
        df = pd.read_csv("processed_spotify_data.csv")
        self.assertTrue(df['valence'].max() <= 1.0)
        self.assertTrue(df['energy'].min() >= 0.0)

if __name__ == '__main__':
    unittest.main()