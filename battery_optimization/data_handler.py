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

        # Fill the missing values with the previous value
        df.fillna(method="ffill", inplace=True)

        return df

    @staticmethod
    def create_template(
        dataframe: pd.DataFrame,
        column: str,
        grouping_columns: List[str],
        methods: List[str] = ["mean"],
        change_col_names: bool = True,
    ) -> pd.DataFrame:
        """Create template Data Frame.

        Create template for a single column with methods over a group of columns.
        A template is intended as the average/min/max/etc. behaviour of a variable
        grouped by other columns of the dataframe (usually time), e.g. the mean value
        of variable x on the different months of the year.

        Args:
            dataframe (pd.DataFrame): input dataframe
            column (str): name of column of which create template.
            grouping_columns (List[str]):
                columns to be used to create the grouping of the template.
            methods (List[str]):
                method to be used to evaluate the template. Defaults to "mean".
                Options: methods in Computations/Descriptive statistics in
                https://pandas.pydata.org/docs/reference/groupby.html passed as
                strings
            change_col_names (bool, optional):
                if True, change template column name according to method used.
                Defaults to True.

        Returns:
            pd.DataFrame: Data Frame with template.
        """
        df_group = dataframe.groupby(grouping_columns)
        templates_dict = {}
        for met in methods:
            df_group_m = eval(f"df_group.{met}(numeric_only=True)[column]")

            name = f"{column}_tpl_{met}" if change_col_names else column
            templates_dict[name] = df_group_m

        df_group = pd.DataFrame(templates_dict)

        if len(grouping_columns) > 1:
            levels = []
            index_name = ""
            for ic, col in enumerate(grouping_columns):
                levels += [ic]
                index_name += (
                    "{0[" + col + "]:.0f}" if ic == 0 else "-{0[" + col + "]:.0f}"
                )
            df_group.reset_index(level=levels, inplace=True)
            df_group["group_index"] = df_group.agg(index_name.format, axis=1)
            df_group.set_index("group_index", drop=True, inplace=True)
        else:
            df_group.index.rename("group_index", inplace=True)
            df_group[grouping_columns[0]] = df_group.index

        return df_group
