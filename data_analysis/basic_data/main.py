import os
import click

import load_dataset as db
import numpy as np

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

def fetch_basic_metrics(dataset, output_dir='.'):
  transfers = dataset['transfers']

  # Drop rows with missing values in relevant columns
  valid_transfers = transfers[['transfer_fee', 'market_value_in_eur']].dropna()
  valid_transfers = valid_transfers[(valid_transfers['transfer_fee'] != 0) & (valid_transfers['market_value_in_eur'] != 0)]

  # Compute statistics
  max_fee = valid_transfers['transfer_fee'].max()
  min_fee = valid_transfers['transfer_fee'].min()
  avg_fee = valid_transfers['transfer_fee'].mean()
  std_fee = valid_transfers['transfer_fee'].std()
  mse = np.mean((valid_transfers['transfer_fee'] - valid_transfers['market_value_in_eur']) ** 2)
  rmse = np.sqrt(mse)

  # Save results to txt
  with open(os.path.join(output_dir, "transfer_fee_stats.txt"), "w") as file:
      file.write(f"Max transfer_fee: {max_fee}\n")
      file.write(f"Min transfer_fee: {min_fee}\n")
      file.write(f"Average transfer_fee: {avg_fee}\n")
      file.write(f"Std deviation transfer_fee: {std_fee}\n")
      file.write(f"MSE between transfer_fee and market_value_in_eur: {mse}\n")
      file.write(f"RMSE between transfer_fee and market_value_in_eur: {rmse}\n")

  return {
      "max_transfer_fee": max_fee,
      "min_transfer_fee": min_fee,
      "avg_transfer_fee": avg_fee,
      "std_transfer_fee": std_fee,
      "mse_transfer_fee_vs_market_value": mse,
      "rmse_transfer_fee_vs_market_value": rmse
  }

@click.command()
@click.option('--output-dir', default='.', help='Directory to save output files')
def main(output_dir):
  dataset = db.load_dataset()
  check_and_save_null_values(dataset, output_dir)
  fetch_basic_metrics(dataset, output_dir)

if __name__ == "__main__":
  main()
