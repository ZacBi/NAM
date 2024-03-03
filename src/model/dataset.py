from datasets.load import load_dataset
from datasets.arrow_dataset  import Dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import random_split as split


# type: ignore
def getDataset(args) -> tuple[Dataset, Dataset, Dataset]:  # type: ignore
    if args.task_type == 'text2image':
        dataset : DatasetDict= load_dataset(args.data_path)
        # 90% train, 10% test + validation
        train_test_valid = dataset['train'].train_test_split(test_size=0.1)
        # Split the 10% test + valid in half test, half valid
        test_valid = train_test_valid['test'].train_test_split(test_size=0.5)
        # gather everyone if you want to have a single DatasetDict
        return train_test_valid['train'], test_valid['test'], test_valid['train']
