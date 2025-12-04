import csv
from torch.utils.tensorboard import SummaryWriter

csv_file_path = 'petnet_fine_tune/training_log.csv'
log_dir = 'petnet_fine_tune/'
writer = SummaryWriter(log_dir=log_dir)

with open(csv_file_path, mode='r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        epoch = int(row[0])
        writer.add_scalar('Train/Loss', float(row[1]), epoch)
        writer.add_scalar('Val/Accuracy', float(row[3]), epoch)
        writer.add_scalar('Val/Loss', float(row[2]), epoch)
        writer.add_scalar('Train/LR', float(row[4]), epoch)

writer.close()
print(f"TensorBoard event files written to: {log_dir}")
