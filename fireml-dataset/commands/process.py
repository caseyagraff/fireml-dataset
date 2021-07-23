import json

from dagma import QueueRunner

from fireml.data.pipeline.dataset_pipeline import get_dataset
from fireml.helpers.parameters import ParameterPrinter


def process(params):
    """Run data processing pipeline."""

    print("=== Params ===")
    print(ParameterPrinter().pprint(params))
    print("==============")

    dataset = get_dataset(params)
    dataset = QueueRunner(dataset)

    return dataset.value
