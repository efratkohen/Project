
from files_process_save import *

import pathlib
import pytest



def test_valid_input():
    fname = pathlib.Path(__file__)
    q = check_file(fname)
    assert fname == q


def test_str_input():
    q = check_file(__file__)
    assert pathlib.Path(__file__) == q


def test_missing_file():
    fname = pathlib.Path('test.fd')
    with pytest.raises(ValueError):
        check_file(fname)


def test_wrong_input_type():
    fname = 2
    with pytest.raises(TypeError):
        check_file(pathlib.Path(fname))


# def test_invalid_input_data_reactor_SV():
#     typeerror_inputs=[-1.0 , 6.0, None, 's']
