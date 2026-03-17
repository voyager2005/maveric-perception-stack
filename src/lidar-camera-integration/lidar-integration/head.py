import json
import os

JSON_DIR = "/home/cv/Documents/points/Data/v1.0-mini/v1.0-mini"

files_to_check = [
    "sample.json", 
    "sample_data.json", 
    "calibrated_sensor.json", 
    "sensor.json", 
    "ego_pose.json"
]

for filename in files_to_check:
    filepath = os.path.join(JSON_DIR, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
        print(f"\n{'='*40}")
        print(f"📄 {filename} (Total records: {len(data)})")
        print(f"{'='*40}")
        # Print the first item nicely formatted
        if len(data) > 0:
            print(json.dumps(data[0], indent=4))
        else:
            print("Empty file!")
