
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


def split_microscopic_to_reactor_len():
    length = 4
    micro_path = check_file('microscopic_data.csv')
    data_microscopic = read_data(micro_path)
    assert len(data_microscopic) == length


def split_microscopic_to_reactor_column_name():
    column_names = [
            "date",
            "ameoba_arcella",
            "ameoba_nude ameba",
            "crawling ciliates_aspidisca",
            "crawling ciliates_trachelopylum",
            "free swimming ciliates_lionutus",
            "free swimming ciliates_paramecium",
            "stalked ciliate_epistylis",
            "stalked ciliate_vorticella",
            "stalked ciliate_carchecium",
            "stalked ciliate_tokophyra",
            "stalked ciliate_podophyra",
            "stalked ciliate_opercularia",
            "rotifer_rotifer",
            "worms_nematode",
            "flagellates_peranema trich",
            "flagellates_micro flagellates",
            "worms_worms",
            "spirochaetes_spirochaetes",
            "Total Count- amoeba",
            "Total Count- Crawling Ciliates",
            "Total Count- Free swimming Ciliates",
            "Total Count- Stalked Ciliates",
            "Total Count- Rotifers",
            "Total Count- Worms",
            "Total Count- Spirochaetes",
            "Total Count- Flagellats",
            "Total Count- Filaments",
            "Filaments_Nocardia_index",
            "Filaments_Microthrix_index",
            "Filaments_N. Limicola_index",
            "Filaments_Thiothrix_index",
            "Filaments_0041/0675_index",
            "Filaments_0092_index",
            "Filaments_1851_index",
            "Filaments_beggiatoa_index",
            "Filaments_zoogloea_index",
        ]
    micro_path = check_file('microscopic_data.csv')
    data_microscopic = read_data(micro_path)
    assert data_microscopic.columns == column_names

# def test_invalid_input_data_reactor_SV():
#     typeerror_inputs=[-1.0 , 6.0, None, 's']
