# Install dependencies as needed:
# pip install kagglehub
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

def load_fifa_data() -> pd.DataFrame:
    """Load the EA FC 24 dataset"""
    file_path = "male_players.csv"
    
    # Load the latest version
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "stefanoleone992/ea-sports-fc-24-complete-player-dataset",
        file_path,
        # Additional arguments can be provided here
    )
    return df

  
# print(load_fifa_data())