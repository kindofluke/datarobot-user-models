"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
"""
In this example we show a complex pipeline with a binary linear model.
"""
import pandas as pd


def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.

    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.

    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model

    Returns
    -------
    pd.DataFrame
    """
    # Execute any steps you need to do before scoring
    # Remove target columns if they're in the dataset
    for target_col in ["Id", "Species"]:
        if target_col in data:
            data.pop(target_col)
    data = data.fillna(0)
    return data


# NOTE: In this model template DRUM is automatically loading "binary_iris.onnx"
#
# Some hooks are omitted here as they are implicitly generated by the DRUM library.
# This happens because DRUM knows how to work with many known types of models.
# They can be overriden in order to change the default behavior:
#
# def load_model(code_dir):
#     ...
#
# def score(data, model, **kwargs):
#     ...
#
# See more about the built-in support of models:
# https://github.com/datarobot/datarobot-user-models/blob/master/DEFINE-INFERENCE-MODEL.md#built-in-model-support
