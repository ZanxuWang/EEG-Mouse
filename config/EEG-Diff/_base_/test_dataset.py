# from EEG import EEGDataset

# test_dataset = dict(
#     type=EEGDataset,
#     #csv_path="data/0310test.csv",  #不对，去掉即可
#     csv_path="data/PN00_1_train.csv"
# )

# # val_dataset = dict(
# #     type=VentilationDataset,
# #     csv_path="data/val_1.csv",
# # )


from EEG import EEGDataset

test_dataset = dict(
    type=EEGDataset,
    csv_path="data/test.csv",
)
