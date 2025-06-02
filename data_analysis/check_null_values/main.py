import os
import click

import load_dataset as db

def check_and_save_null_values(dataset, output_dir):
  os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
  with open(os.path.join(output_dir, "null_values_report.txt"), "w") as file:
    for table_name, table in dataset.items():
      total_entries = len(table)
      file.write(f"Null values in {table_name} (Total entries: {total_entries}):\n")
      null_counts = table.isnull().sum()
      for column, count in null_counts.items():
        file.write(f"  {column}: {count} null values\n")
      file.write("\n")

@click.command()
@click.argument('output_dir')
def main(output_dir):
  dataset = db.load_dataset()
  check_and_save_null_values(dataset, output_dir)

if __name__ == "__main__":
  main()
