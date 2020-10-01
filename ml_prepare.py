import clean_data_svi as cds

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime


class ML_prepare:
    """
    This class is based on the cleaned raw data tables in the "clean_tables" folder.

    This class allows for a quick, simple generation of the data
    defined by the desired hyper paramater values: 
    1) Day of delay (between the microscopic observation and water results)
    2) Section of organisms to test ('all','filaments','various','total counts')

    The user can easily get a table that represents the "X" and "y" he chooses, 
    ready for testing with Machine Learning models.
    There is also an option to chooses whether the "y" is the labeled data or the raw values.

    attributes
    --------
    _svi_lst, _micro_lst: list
        list of DF, read from the cleaned tables
    _delay: int
        number of days delayed between the microscopic observation and sv/svi results.
    _x, _y: pd.DataFrame
        contain the whole data, of all 4 bio_reactors, not yet devided to sections, 
        and still containing the pre-existing NaN values
    delay_table: pd.DataFrame
        x and y concatenated, with keys "micro" and "svi" as extra index level.
    """

    def __init__(self, delay: int):
        self._svi_lst = self.__read_and_index_svi_tables()
        self._micro_lst = self.__read_and_index_micro_tables()
        self._delay = delay

        self._x, self._y = self.__create_x_y_delayed()
        self.delay_table = self.__join_x_y()  # with partial NaN rows un-touched.

    @property
    def svi_lst(self):
        return self._svi_lst

    @property
    def micro_lst(self):
        return self._micro_lst

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def delay(self):
        return self._delay

    def __read_and_index_svi_tables(self):
        '''
        Reads 4 svi tables using helper function.
        Sets datetime index using helper function.

        return
        -------
        svi_tables: list of DF
        '''
        svi_tables = self.__read_clean_tables("svi")
        cds.set_datetime_index(svi_tables)
        return svi_tables

    def __read_and_index_micro_tables(self):
        '''
        Reads 4 microscopic tables using helper function.
        Sets datetime index using helper function.

        return
        -------
        micro_tables: list of DF
        '''
        micro_tables = self.__read_clean_tables("micro")
        cds.set_datetime_index(micro_tables)
        return micro_tables

    def __read_clean_tables(self, data_type: str):
        '''
        Reads 4 data tables from files, to list of DF.

        Parameters
        ----------
        data_type: str
            "micro" / "svi"

        return
        ---------
        clean_tables_lst: list of DF
        '''
        assert data_type in {"micro", "svi"}, '"data_type" must be "micro" / "svi"'
        clean_tables_lst = []
        for i in range(4):
            table = pd.read_csv(f"clean_tables/{data_type}_{i+1}.csv", index_col=False)
            clean_tables_lst.append(table)
        return clean_tables_lst

    def get_columns_of_sections(self):
        '''
        Creates lists of column names for each group of organisms

        return
        ---------
        total_cols: list
        filament_cols: list
        various_organisms_cols: list
        '''
        total_cols = [col for col in self._x.columns if "Total" in col]
        filament_cols = [col for col in self._x.columns if "Filaments_" in col]
        various_organisms_cols = [
            col
            for col in self._x.columns
            if "Filaments_" not in col and "Total" not in col
        ]
        return total_cols, filament_cols, various_organisms_cols

    def get_partial_table(self, x_section: str, y_labels: bool = False):
        """
        Generates a table with only the desired groups of organisms.
        Option to choose labeled or non-labeled data.

        Parameters
        -----------
        x_section: str
            'all' / 'total_counts' / 'filaments' / 'various'
            * 'all' - gives all single organisms (excludeing total counts)
        y_labels: bool
            determines if results in returned table will be labeled

        return
        --------
        ready_xy_table: pd.DataFrame
            desired table, no NaN values, with keys "x" and "y" as extra index level.
        """
        assert x_section in {"all", "total_counts", "filaments", "various"}, (
            "x_section invalid."
            + "expected 'all' / 'total_counts' / 'filaments' / 'various'"
        )

        # get column names
        (
            total_cols,
            filament_cols,
            various_organisms_cols,
        ) = self.get_columns_of_sections()

        # get x groups as desired
        if x_section == "all":  # no total counts
            various_x = self._x.loc[:, various_organisms_cols].reset_index(
                level=1, drop=True
            )
            filaments_x = self._x.loc[:, filament_cols].reset_index(level=1, drop=True)
            only_x = pd.concat([various_x, filaments_x], axis=1)
        elif x_section == "total_counts":
            only_x = self._x.loc[:, total_cols].reset_index(level=1, drop=True)
        elif x_section == "filaments":
            only_x = self._x.loc[:, filament_cols].reset_index(level=1, drop=True)
        elif x_section == "various":
            only_x = self._x.loc[:, various_organisms_cols].reset_index(
                level=1, drop=True
            )

        # get chosen y data
        if y_labels:
            only_y = self._y.loc[:, "SV_label":"SVI_label"].reset_index(
                level=1, drop=True
            )
        else:
            only_y = self._y.loc[:, "Settling_velocity":"SVI"].reset_index(
                level=1, drop=True
            )

        # concat x and y
        ready_xy_table = pd.concat([only_x, only_y], keys=["x", "y"], axis=1)
        ready_xy_table.dropna(inplace=True)

        return ready_xy_table


    def plot_svi(self):
        '''
        Plot SV and SVI results over the time period in the raw data.
        '''
        fig_svi, axes = plt.subplots(2, 1)
        fig_svi.suptitle("SV and SVI per reactors")
        for i in range(4):
            axes[0].plot(self.svi_lst[i]["SVI"], label=f"bio reactor {i+1}")
            axes[1].plot(self.svi_lst[i]["Settling_velocity"])
        axes[0].set_ylabel("SVI")
        axes[1].set_ylabel("SV")
        axes[0].set_xticks([])
        axes[0].legend()
        plt.xticks(rotation=70)
        plt.show()

    def __create_x_y_delayed(self):
        """
        joins all 4 bioreactor tables to create raw data with the desired delay in 
        days between x (microscopic data) and y (sv/svi data).
        Using helper function to create the delay of every bioreactor before joining. 
        The resulting DF has an extra level of columns named '1','2','3','4'
        to access each bioreactor.
        
        return
        ------
        - x: pd.DataFrame
            microscopic data, all 4 bio-reactors one after the other
        - y: pd.DataFrame
            sv/svi data, all 4 bio-reactors one after the other
        """
        svi_y_lst = []
        micro_x_lst = []
        for bio_reactor_i in range(4):  # loop over bio reactors
            micro_x, svi_y = self.__create_x_y_bioreactor(bio_reactor_i)
            micro_x_lst.append(micro_x)
            svi_y_lst.append(svi_y)

        x = pd.concat(
            [micro_x_lst[0], micro_x_lst[1], micro_x_lst[2], micro_x_lst[3]],
            keys=["1", "2", "3", "4"],
            names=["bio_reactor", "date"],
        )
        y = pd.concat(
            [svi_y_lst[0], svi_y_lst[1], svi_y_lst[2], svi_y_lst[3]],
            keys=["1", "2", "3", "4"],
            names=["bio_reactor", "date"],
        )

        return x, y

    def __create_x_y_bioreactor(self, bio_reactor_i: int):
        """
        Creates the delay for this bioreactor between x (microscopic data) and y (sv/svi data).
        Using helper function to find the matching date for every row.
        
        Parameters
        ---------
        bio_reactor_i: int
            # of bioreactor
        
        return 
        --------
        micro_x: pd.DataFrame
            microscopic data for this bioreactor
        svi_y: pd.DataFrame
            sv/svi data for this bioreactor, with matching date rows according to delay
        """
        svi_y = pd.DataFrame(columns=self._svi_lst[0].columns)  # empty svi_y
        micro_x = self._micro_lst[bio_reactor_i].copy()
        matching_dates = []
        for date in self._micro_lst[bio_reactor_i].index:
            closest_date = self._find_closest_date(bio_reactor_i, date)
            if not closest_date:  # if this is already out of bounds
                # remove all rows from this point on from micro_x:
                micro_x.drop(micro_x.loc[date:].index, inplace=True)
                break
            else:
                matching_dates.append(closest_date)

        # add all rows of desired dates:
        svi_y = pd.concat([svi_y, self._svi_lst[bio_reactor_i].loc[matching_dates]])

        assert (
            svi_y.shape[0] == micro_x.shape[0]
        ), f"x and y for bio reactor {bio_reactor_i} not same length"
        return micro_x, svi_y


    def _find_closest_date(self, bio_reactor_i: int, date0: datetime):
        """
        Finds in the DF svi[bio_reactor_i] the closest row with date = date0 + delay
        If it is after the last date, return False.
        If there is no such row, goes to the closest previous row.

        Parameters
        ---------
        bio_reactor_i: int
            # of bioreactor
        date0: datetime
            data of row in micro_df to find the matching row in svi_df
        
        return 
        --------
        final_date: datetime
        """
        delay_time = timedelta(days=self._delay)
        final_date = date0 + delay_time

        # if out of bounds
        last_date = self._svi_lst[bio_reactor_i].index[-1]
        if final_date > last_date:
            return False

        # else:
        while True:
            if final_date in self._svi_lst[bio_reactor_i].index:
                return final_date
            else:  # go one date back
                final_date -= timedelta(days=1)


    def __join_x_y(self):
        '''
        Concatenates raw data x and y, with keys "micro" and "svi" as extra index level.

        return
        --------
        concatenated table: pd.DataFrame
        '''
        x = self._x.reset_index(level=1, drop=True)
        y = self._y.reset_index(level=1, drop=True)
        return pd.concat([x, y], axis=1, keys=["micro", "svi"])


if __name__ == "__main__":
    delay = 4
    data = ML_prepare(delay)
    data.plot_svi()

    delay_lst = [*range(0, 18, 3)]
    sections = ["all", "total_counts", "filaments", "various"]

    t = data.get_partial_table(x_section="all", y_labels=True)
    t1 = data.get_partial_table(x_section="total_counts", y_labels=True)
    t2 = data.get_partial_table(x_section="filaments", y_labels=False)
    t3 = data.get_partial_table(x_section="various", y_labels=False)

