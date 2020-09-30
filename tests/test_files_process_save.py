
from files_process_save import *
import clean_data_svi as cds
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


def test_split_microscopic_to_reactor_len():
    length = 4
    micro_path = check_file('microscopic_data.csv')
    data = read_data(micro_path)
    data_microscopic = split_microscopic_to_reactor(data)
    assert len(data_microscopic) == length


def test_split_microscopic_to_reactor_column_name():
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
    data = read_data(micro_path)
    data_microscopic = split_microscopic_to_reactor(data)
    assert list(data_microscopic[0].columns) == column_names


def test_split_svi_to_reactor_column_names():
    columns_names = ['date', 'Settling_velocity', 'SVI']
    svi_path = check_file('SVI.csv')
    data = read_data(svi_path)
    data_svi_computed = cds.svi_calculate(data)
    data_svi = split_svi_to_reactor(data_svi_computed)
    assert list(data_svi[0].columns) == columns_names


def test_split_svi_to_reactor_len():
    length = 4
    svi_path = check_file('SVI.csv')
    data = read_data(svi_path)
    data_svi_computed = cds.svi_calculate(data)
    data_svi = split_svi_to_reactor(data_svi_computed)
    assert len(data_svi) == length


def test_clean_table_SVI_dates():
    days_list= []
    length_list = []
    for i in range(4):
        svi_path = check_file(f"clean_tables/svi_{i+1}.csv")
        data = pd.read_csv(svi_path)
        data["date"] = pd.to_datetime(data["date"], dayfirst=True)
        df = data["date"]
        length_list.append(len(df))
        days = df.diff().sum().days
        days_list.append(days+1)
    assert days_list == length_list
