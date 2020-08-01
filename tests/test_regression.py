from ml_prepare import ML_prepare
from regression import *

import pytest


def test_create_models_dict():
    models_dict = create_models_dict()
    assert isinstance(models_dict, dict)
    # assert models are from sklearn module
    for model in models_dict:
        module = getattr(model, '__module__', None)
        assert 'sklearn' in module


def test_get_scores_of_all_models():
    models_dict = create_models_dict()
    model_names = set(models_dict.values())
    delay_range = range(4, 11)
    scores_models_dict = get_scores_of_all_models(models_dict, delay_range, print_flag=False)
    assert isinstance(scores_models_dict, dict)
    for key in scores_models_dict:
        assert key in model_names


def test_get_scores_of_model():
    models_dict = create_models_dict()
    [regr_model, model_name] = list(models_dict.items())[0]

    delay_range = range(4, 11)
    res = get_scores_of_model(regr_model, model_name, delay_range, print_flag=False)
    assert isinstance(res, dict)
    assert set(res.keys())==set(delay_range)


def test_loop_over_sections_and_y():
    data = ML_prepare(7)
    models_dict = create_models_dict()
    regr_model = list(models_dict.keys())[0]
    res = loop_over_sections_and_y(data, regr_model)
    assert isinstance(res, tuple)
    assert len(res)==9


def test_regr_model_func():
    # create data
    # get_partial_data
    # get x, get y
    # score, fitted_model = regr_model_func(X, y, reg_model)
    # assert score is float
    # assert fitted_model is sklearn:
        # module = getattr(las, '__module__', None)
        # assert 'sklearn' in module
    pass


def test_create_list_of_tidy_df_by_day_types():
    # create data
    # get model_dict
    # create delay_range = range(1, 13)
    # scores_models_dict = get_scores_of_all_models(models_dict, delay_range=delay_range, print_flag=False)
    # days_df_dict = create_list_of_tidy_df_by_day(scores_models_dict, delay_range)
    # assert type is dict
    # assert len==range
    # x = days_df_dict[i]
    # assert type(x) is df
    pass


def test_create_list_of_tidy_df_by_day_values():
    # create data
    # get model_dict
    # create delay_range = range(1, 13)
    # scores_models_dict = get_scores_of_all_models(models_dict, delay_range=delay_range, print_flag=False)
    # days_df_dict = create_list_of_tidy_df_by_day(scores_models_dict, delay_range)
    # x = days_df_dict[i]
    # assert x.shape == (40,4)
    # assert all(x['model'].value_counts()==8)
    # assert all(x['sv_svi'].value_counts()==20)
    # assert all(x['section'].value_counts()==10)
    pass


def test_get_day3_filaments_svi_data():
    # filaments_x, filaments_svi = get_day3_filaments_svi_data()
    # assert isinstance(filaments_x, pd.DataFrame)
    # assert isinstance(filaments_svi, pd.Series)
    # assert filaments_x.shape[0]==filaments_svi.shape[0]
    # assert filaments_x.shape[1]==9 # num of filament columns
    # filaments_svi.name[1]=='SVI'
    pass


def test_display_weights_of_winning_model():
    # get models_dict
    # winning_model = list(models_dict.keys())[0]
    # df_coefs = display_weights_of_winning_model(winning_model)
    # assert isinstance(df_coefs, pd.DataFrame)
    # assert df_coefs.shape==(9,2)
    pass
