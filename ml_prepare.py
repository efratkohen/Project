import pandas as pd
from datetime import datetime, timedelta
import clean_data_svi as cds
import matplotlib.pyplot as plt


class ML_prepare:
    def __init__(self, delay: int):
        self._svi_lst = self.__read_and_index_svi_tables()
        self._micro_lst = self.__read_and_index_micro_tables()
        self._delay = delay

        self._x, self._y = self.__create_x_y_delayed(days_delay=self._delay)
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

    def get_partial_table(self, x_section: str, y_labels: bool = False):
        """
        x_section: str. 'all' / 'total_counts' / 'filaments' / 'various'

        'all' - gives all single organisms, (excludes total counts)
        """
        assert x_section in {"all", "total_counts", "filaments", "various"}, (
            "x_section invalid."
            + "expected 'all' / 'total_counts' / 'filaments' / 'various'"
        )

        total_cols = [col for col in self._x.columns if "Total" in col]
        filament_cols = [col for col in self._x.columns if "Filaments_" in col]
        various_organisms_cols = [
            col
            for col in self._x.columns
            if "Filaments_" not in col and "Total" not in col
        ]

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

        if y_labels:
            only_y = self._y.loc[:, "SV_label":"SVI_label"].reset_index(
                level=1, drop=True
            )
        else:
            only_y = self._y.loc[:, "Settling_velocity":"SVI"].reset_index(
                level=1, drop=True
            )

        ready_xy_table = pd.concat([only_x, only_y], keys=["x", "y"], axis=1)
        ready_xy_table.dropna(inplace=True)

        return ready_xy_table

    def plot_svi(self):
        fig_svi, axes = plt.subplots(2, 1)
        fig_svi.suptitle("SV and SVI per reactors")
        for i in range(4):
            axes[0].plot(self.svi_lst[i]["SVI"], label=f"bio reactor {i+1}")
            axes[1].plot(self.svi_lst[i]["Settling_velocity"])
        axes[0].set_ylabel("SVI")
        axes[0].set_xticks([])
        axes[0].legend()
        plt.xticks(rotation=70)
        plt.show()

    def __read_and_index_svi_tables(self):
        svi_tables = self.__read_clean_tables("svi")
        cds.set_datetime_index(svi_tables)
        return svi_tables

    def __read_and_index_micro_tables(self):
        micro_tables = self.__read_clean_tables("micro")
        cds.set_datetime_index(micro_tables)
        return micro_tables

    def __read_clean_tables(self, data_type: str):
        assert data_type in {"micro", "svi"}, '"data_type" must be "micro" / "svi"'
        clean_tables_lst = []
        for i in range(4):
            table = pd.read_csv(f"clean_tables/{data_type}_{i}.csv", index_col=False)
            clean_tables_lst.append(table)
        return clean_tables_lst

    def __create_x_y_delayed(self, days_delay: int):
        """
        Returns:
        ------
        - micro_x - df of all 4 bio-reactors one after the other
        - svi_y
        """
        self._delay = days_delay

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

        self._x = x
        self._y = y

        return self.x, self.y

    def __join_x_y(self):
        x = self._x.reset_index(level=1, drop=True)
        y = self._y.reset_index(level=1, drop=True)
        return pd.concat([x, y], axis=1, keys=["micro", "svi"])

    def __create_x_y_bioreactor(self, bio_reactor_i):
        """
        Returns for this bio reactor the micro_x and svi_y with the correct delay.
        """
        svi_y = pd.DataFrame(columns=self._svi_lst[0].columns)  # empty svi_y
        micro_x = self._micro_lst[bio_reactor_i].copy()
        matching_dates = []
        for date in self._micro_lst[bio_reactor_i].index:
            closest_date = self.__find_closest_date(bio_reactor_i, date)
            # print(f'date0 = {date}, closest date = {closest_date}') # later
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

    def __find_closest_date(self, bio_reactor_i: int, date0):
        """
        Gets bio_reactor and date, 
        Finds in svi[bio_reactor_i] the closest row with date = date0 + delay
        If it is after the last date, return False
        If there is no such row, goes to the closest previous row.
        """

        delay_time = timedelta(days=self._delay)
        final_date = date0 + delay_time

        # if out of bounds
        last_date = self._svi_lst[bio_reactor_i].index[-1]
        if final_date > last_date:
            # print("out of bounds")  # later
            return False

        # else:
        while True:
            if final_date in self._svi_lst[bio_reactor_i].index:
                return final_date
            else:  # go one date back
                print(
                    f"date {final_date} is not found in svi bio reactor {bio_reactor_i}"
                )  # later
                final_date -= timedelta(days=1)


if __name__ == "__main__":
    delay = 4
    data = ML_prepare(delay)
    # data.plot_svi()

    delay_lst = [*range(0, 18, 3)]
    sections = ["all", "total_counts", "filaments", "various"]

    t = data.get_partial_table(x_section="all", y_labels=True)
    t1 = data.get_partial_table(x_section="total_counts", y_labels=True)
    t2 = data.get_partial_table(x_section="filaments", y_labels=False)
    t3 = data.get_partial_table(x_section="various", y_labels=False)

