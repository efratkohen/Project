from ml_prepare import ML_prepare
from regression import *

import pytest
import pandas as pd


def test_create_models_dict():
    models_dict = create_models_dict()
    assert isinstance(models_dict, dict)
    # assert models are from sklearn module
    for model in models_dict:
        module = getattr(model, "__module__", None)
        assert "sklearn" in module


def test_get_scores_of_all_models():
    models_dict = create_models_dict()
    model_names = set(models_dict.values())
    delay_range = range(4, 11)
    scores_models_dict = get_scores_of_all_models(
        models_dict, delay_range, print_flag=False
    )
    assert isinstance(scores_models_dict, dict)
    for key in scores_models_dict:
        assert key in model_names


def test_get_scores_of_model():
    models_dict = create_models_dict()
    [regr_model, model_name] = list(models_dict.items())[0]

    delay_range = range(4, 11)
    res = get_scores_of_model(regr_model, model_name, delay_range, print_flag=False)
    assert isinstance(res, dict)
    assert set(res.keys()) == set(delay_range)


def test_loop_over_sections_and_y():
    data = ML_prepare(7)
    models_dict = create_models_dict()
    regr_model = list(models_dict.keys())[0]
    res = loop_over_sections_and_y(data, regr_model)
    assert isinstance(res, tuple)
    assert len(res) == 8


def test_regr_model_func():
    data = ML_prepare(1)
    table_xy = data.get_partial_table(x_section="various", y_labels=False)
    X, y = table_xy.loc[:, "x"], table_xy.loc[:, "y"]
    models_dict = create_models_dict()
    regr_model = list(models_dict.keys())[0]
    score, fitted_model = regr_model_func(X, y, regr_model)
    assert isinstance(score, float)

    module = getattr(fitted_model, "__module__", None)
    assert "sklearn" in module


def test_create_list_of_tidy_df_by_day_types():
    data = ML_prepare(4)
    models_dict = create_models_dict()
    delay_range = range(8, 14)
    scores_models_dict = get_scores_of_all_models(
        models_dict, delay_range=delay_range, print_flag=False
    )
    days_df_dict = create_list_of_tidy_df_by_day(scores_models_dict, delay_range)

    assert isinstance(days_df_dict, dict)
    assert set(days_df_dict.keys()) == set(delay_range)
    assert isinstance(days_df_dict[8], pd.DataFrame)


def test_create_list_of_tidy_df_by_day_values():
    data = ML_prepare(4)
    models_dict = create_models_dict()
    delay_range = range(1, 13)
    scores_models_dict = get_scores_of_all_models(
        models_dict, delay_range=delay_range, print_flag=False
    )
    days_df_dict = create_list_of_tidy_df_by_day(scores_models_dict, delay_range)

    df4 = days_df_dict[4]
    assert df4.shape == (40, 4)
    assert all(df4["model"].value_counts() == 8)
    assert all(df4["sv_svi"].value_counts() == 20)
    assert all(df4["section"].value_counts() == 10)


def test_get_day3_filaments_svi_data_types():
    filaments_x, filaments_svi = get_day3_filaments_svi_data()
    assert isinstance(filaments_x, pd.DataFrame)
    assert isinstance(filaments_svi, pd.Series)


def test_get_day3_filaments_svi_data_shapes():
    filaments_x, filaments_svi = get_day3_filaments_svi_data()
    assert filaments_x.shape[0] == filaments_svi.shape[0]
    assert filaments_x.shape[1] == 9  # num of filament columns
    assert filaments_svi.name[1] == "SVI"


def test_display_weights_of_winning_model():
    models_dict = create_models_dict()
    some_model = list(models_dict.keys())[0]
    df_coefs = display_weights_of_winning_model(some_model)
    assert isinstance(df_coefs, pd.DataFrame)
    assert df_coefs.shape == (9, 2)  # num of filament
