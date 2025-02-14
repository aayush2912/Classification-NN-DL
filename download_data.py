import kaggle

kaggle.api.dataset_download_files("agrigorev/clothing-dataset-full", path="./clothing_dataset", unzip=True)

print("Dataset downloaded!")
