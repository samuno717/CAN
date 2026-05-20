from tqdm import tqdm
import os
from config import DATA_DIR
import argparse

ids = set()
first_log_timestamp = None
final_log_timestamp = None
previous_time_for_id = {}

TOTAL_LINES = 5910615

parser = argparse.ArgumentParser(description="CAN raw frame parser.")
parser.add_argument("filename", type=str, help="File with raw CAN data, for example: candump-2026-04-02_201058.log")
args = parser.parse_args()

log_file_path = DATA_DIR / args.filename
csv_file_path = os.path.join(DATA_DIR, 'processed_data.csv')

with open(log_file_path, 'r') as logfile, open(csv_file_path, 'w') as csvfile:
    csvfile.write("Timestamp,delta_t,CAN_ID,Byte1,Byte2,Byte3,Byte4,Byte5,Byte6,Byte7,Byte8\n")

    for l in tqdm(logfile, total=TOTAL_LINES, desc="Processing frames", unit=" frame"):
        raw_line = l.strip().split()
        raw_line.pop(1)

        current_timestamp = float(raw_line[0].strip("()"))

        if first_log_timestamp is None:
            first_log_timestamp = current_timestamp
        final_log_timestamp = current_timestamp

        raw_can_id = raw_line[1].split("#")[0]
        can_id = int(raw_can_id, 16)
        ids.add(can_id)

        raw_payload = raw_line[1].split("#")[1]
        split_payload = [raw_payload[i:i+2] for i in range(0, len(raw_payload), 2)]
        payload = [int(i, 16) for i in split_payload]

        if can_id in previous_time_for_id:
            delta_t = current_timestamp - previous_time_for_id[can_id]
        else:
            delta_t = 0.0

        previous_time_for_id[can_id] = current_timestamp
        delta_t = round(delta_t, 6)

        payload_str = ",".join(str(b) for b in payload)

        csvline = f"{current_timestamp},{delta_t},{can_id},{payload_str}\n"
        csvfile.write(csvline)

print("\nDONE")
print("File created: processed_data.csv in data folder.")

ecu_count = len(ids)
total_time = final_log_timestamp - first_log_timestamp



print("\n" + "-" * 30)
print(f"Unique ECUs: {ecu_count}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Total frames: {TOTAL_LINES}")
print("-" * 30)