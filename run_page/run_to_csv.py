import sqlite3

try:
    import pandas as pd
except Exception:
    raise Exception("please install pandas run: pip3 install pandas")
from math import floor

data = sqlite3.connect("run_page/data.db")
df = pd.read_sql_query("SELECT * FROM activities where type = 'Run' ", data)

def apply_duration_time(d):
    try:
        return d.split()[1].split(".")[0]
    except Exception as e:
        print(f"Error applying duration time: {e}")
        return ""


# we do not need polyline in csv
# df = df.drop("summary_polyline", axis=1)
df["elapsed_time"] = df["elapsed_time"].apply(apply_duration_time)
df["moving_time"] = df["moving_time"].apply(apply_duration_time)

# Round distance column to 1 decimal place
if "distance" in df.columns:
    df["distance"] = df["distance"].round(1)


def format_pace(d):
    """
    Convert speed (m/s) to pace format.

    Args:
        d: Speed in meters per second (m/s)

    Returns:
        str: Pace in format "{minutes}''{seconds}" (minutes per kilometer, min/km)
             Example: "5''30" means 5 minutes 30 seconds per kilometer
    """
    if not d:
        return "0"
    pace = (1000.0 / 60.0) * (1.0 / d)
    minutes = floor(pace)
    seconds = floor((pace - minutes) * 60.0)
    return f"{minutes}分{seconds}秒/公里"

df["average_speed"] = df["average_speed"].apply(format_pace)
df = df.sort_values(by=["start_date"])

# Write to CSV without index
csv_path = "assets/run.csv"
df.to_csv(csv_path, index=False, lineterminator='\n')

# Remove trailing newline from the file
with open(csv_path, 'rb+') as f:
    f.seek(-1, 2)  # Go to the last byte
    if f.read(1) == b'\n':
        f.seek(-1, 2)  # Go back one byte
        f.truncate()  # Remove the last byte
