# from EEG import EEGDataset,evaluationDataset

# train_dataset = dict(
#     type= evaluationDataset, #EEGDataset,
#     #csv_path="data/SCD_train.csv", 
#     csv_path="data/PN00_1_train.csv"
# )

# val_dataset = dict(
#     type=EEGDataset,
#     #csv_path="data/SCD_train.csv",
#     csv_path="data/PN00_1_train.csv"
# )


from EEG import EEGDataset

train_dataset = dict(
    type=EEGDataset,
    csv_path="data/train.csv",
)

val_dataset = dict(
    type=EEGDataset,
    csv_path="data/test.csv",
)
