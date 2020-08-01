from ml_prepare import ML_prepare
from regression import *

import pytest


def test_create_models_dict():
    # assert type dict
    # loop over dic keys():
        # module = getattr(las, '__module__', None)
        # assert 'sklearn' in module
    pass


def test_get_scores_of_all_models():
    # get model_dict
    # create delay_range = range(1, 13)
    # get_scores_of_all_models(models_dict, delay_range, print_flag=False)
    # assert type dict
    # assert keys are 5 models
    pass


def test_get_scores_of_model():
    # create delay_range
    # get one model, model_dict[1]
    # res = get_scores_of_model(regr_model, model_name: str, delay_range, print_flag=False)
    # assert type dict
    # assert keys fit range
    pass


def test_loop_over_sections_and_y():
    # create data
    # res = loop_over_sections_and_y(data, regr_model)
    # assert res is namedtuple
    # assert res len 8


def test_regr_model_func():
    # create data
    # get_partial_data
    # get x, get y
    # score, fitted_model = regr_model_func(X, y, reg_model)
    # assert score is float
    # assert fitted_model is sklearn:
        # module = getattr(las, '__module__', None)
        # assert 'sklearn' in module


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


def test_get_day3_filaments_svi_data():
    # filaments_x, filaments_svi = get_day3_filaments_svi_data()
    # assert isinstance(filaments_x, pd.DataFrame)
    # assert isinstance(filaments_svi, pd.Series)
    # assert filaments_x.shape[0]==filaments_svi.shape[0]
    # assert filaments_x.shape[1]==9 # num of filament columns
    # filaments_svi.name[1]=='SVI'


def test_display_weights_of_winning_model():
    # get models_dict
    # winning_model = list(models_dict.keys())[0]
    # df_coefs = display_weights_of_winning_model(winning_model)
    # assert isinstance(df_coefs, pd.DataFrame)
    # assert df_coefs.shape==(9,2)
    