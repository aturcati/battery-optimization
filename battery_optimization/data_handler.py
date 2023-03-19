import logging
from pathlib import Path
from typing import List, Optional

import holidays
import pandas as pd

from battery_optimization.const import DATA_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


class DataHandler(object):
    """
    DataHandler is a utility class that handles the data loading and processing.
    """

    def __init__(self, reprocess: bool = False):
        # Check if the processed file exists, and load it, if it does not exist, process the csv file first
        processed_filenames = {
            "Day-ahead_Prices_60min.csv": "dayahead_hourly.feather",
            "Day-ahead_Prices_15min.csv": "dayahead_fifteen.feather",
        }
        processed_df = {}
        for f, pf in processed_filenames.items():
            if not Path(DATA_DIR / pf).is_file() or reprocess:
                logging.info(f"Processing csv file: {f}")
                try:
                    df = self.process_csv(f)
                    df.to_feather(DATA_DIR / pf)
                except Exception as e:
                    logging.exception(e)

            logging.info(f"Loading existing processed file: {pf}")
            try:
                processed_df[pf] = pd.read_feather(DATA_DIR / pf)
            except Exception as e:
                logging.exception(e)

        # Load the processed data into the class
        self.hourly_df = processed_df["dayahead_hourly.feather"]
        self.fifteen_df = processed_df["dayahead_fifteen.feather"]

    @staticmethod
    def process_csv(filename: str) -> pd.DataFrame:
        """
        process_csv processes the csv file and returns a pandas DataFrame with the cleaned data.

        Args:
            filename (str): name of the csv file in the data directory

        Returns:
            pd.DataFrame: dataframe with the cleaned data from the csv file with columns [Day-ahead Price [EUR/MWh], start_mtu, end_mtu]
        """
        df = pd.read_csv(DATA_DIR / filename)

        # Convert the start and end mtu to datetime
        df["start_mtu"] = pd.to_datetime(
            df["MTU (CET/CEST)"].apply(lambda x: x.split(" - ")[0].strip()),
            format="%d.%m.%Y %H:%M",
        )
        df["end_mtu"] = pd.to_datetime(
            df["MTU (CET/CEST)"].apply(lambda x: x.split(" - ")[0].strip()),
            format="%d.%m.%Y %H:%M",
        )

        # Add time features
        df["year"] = df["start_mtu"].dt.year
        df["month"] = df["start_mtu"].dt.month
        df["week_of_year"] = df["start_mtu"].dt.isocalendar().week
        df["day"] = df["start_mtu"].dt.day
        df["hour"] = df["start_mtu"].dt.hour
        df["minute"] = df["start_mtu"].dt.minute
        df["day_of_week"] = df["start_mtu"].dt.dayofweek
        df["day_of_month"] = df["start_mtu"].dt.day
        df["day_name"] = df["start_mtu"].dt.day_name()

        # Add holiday feature
        de_holidays = holidays.country_holidays("DE")
        df["is_holiday"] = [d in de_holidays for d in df["start_mtu"].dt.date]

        # Rename the price column and drop unnecessary columns
        df.rename(columns={"Day-ahead Price [EUR/MWh]": "price"}, inplace=True)
        df.drop(columns=["MTU (CET/CEST)", "Currency", "BZN|DE-LU"], inplace=True)

        # Add normalized price column
        df["normalized_price"] = (df["price"] - df["price"].mean()) / df["price"].std()

        # Fill the missing values due to start of Daylight saving times with the previous value
        df.fillna(method="ffill", inplace=True)

        return df
