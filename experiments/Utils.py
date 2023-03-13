import pandas as pd


class Utils:

    @staticmethod
    def addChange(df: pd.core.frame.DataFrame, column: str) -> pd.core.frame.DataFrame:
        "The method calculates pct change <0;1> of the given column. "
        df["Pct change of " + column] = df[column].pct_change()
        return df
    
    @staticmethod
    def addNormalizedColumn(df: pd.core.frame.DataFrame, column: str) -> pd.core.frame.DataFrame:
        "The method calculates normalized value <0;1> of the given column. "
        df["Normalized " + column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df

    @staticmethod
    def load_settings(filepath):
        settings = {}
        with open(filepath) as f:
            for line in f:
                key, value = line.strip().split('=')
                value = value.strip()
                if value.isdigit():
                    settings[key.strip()] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    settings[key.strip()] = float(value)
                elif value.lower() == 'true':
                    settings[key.strip()] = True
                elif value.lower() == 'false':
                    settings[key.strip()] = False
                else:
                    settings[key.strip()] = value
        return settings