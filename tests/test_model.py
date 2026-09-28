import numpy as np
import pandas as pd

from diabetes_model import FEATURES, ZERO_AS_MISSING, build_preprocessor, load_data


def test_load_data_has_no_duplicates():
    X, y = load_data()
    assert not pd.concat([X, y], axis=1).duplicated().any()


def test_zeros_are_imputed_not_kept():
    X, _ = load_data()
    transformed = build_preprocessor().fit(X).transform(X)
    out = pd.DataFrame(transformed, columns=build_preprocessor().fit(X).get_feature_names_out())
    assert not out.isna().any().any()
    assert (out[ZERO_AS_MISSING] > 0).all().all()


def test_preprocessor_keeps_all_features():
    X, _ = load_data()
    names = set(build_preprocessor().fit(X).get_feature_names_out())
    assert names == set(FEATURES)
