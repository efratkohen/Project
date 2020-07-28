import pandas as pd
from datetime import datetime, timedelta
import files_process_save as fps
import matplotlib.pyplot as plt


class ml_prepare:
    def __init__(self):
        self._svi_lst = self.read_and_index_svi_tables()
        self._micro_lst = self.read_and_index_micro_tables()
        self._x = 0
        self._y = 0
        self._delay = 0

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

    def read_and_index_svi_tables(self):
        svi_tables = self.__read_clean_tables("svi")
        fps.set_datetime_index(svi_tables)
        return svi_tables

    def read_and_index_micro_tables(self):
        micro_tables = self.__read_clean_tables("micro")
        fps.set_datetime_index(micro_tables)
        return micro_tables

    def __read_clean_tables(self, data_type: str):
        assert data_type in {"micro", "svi"}, '"data_type" must be "micro" / "svi"'
        clean_tables_lst = []
        for i in range(4):
            table = pd.read_csv(f"clean_tables/{data_type}_{i}.csv", index_col=False)
            clean_tables_lst.append(table)
        return clean_tables_lst

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

    def create_x_y_delayed(self, days_delay: int):
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
            micro_x, svi_y = data.create_x_y_bioreactor(bio_reactor_i)
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

    def create_x_y_bioreactor(self, bio_reactor_i):
        """
        Returns for this bio reactor the micro_x and svi_y with the correct delay.
        """
        svi_y = pd.DataFrame(columns=data._svi_lst[0].columns)  # empty svi_y
        micro_x = data._micro_lst[bio_reactor_i].copy()
        matching_dates = []
        for date in data._micro_lst[bio_reactor_i].index:
            closest_date = data.find_closest_date(bio_reactor_i, date)
            # print(f'date0 = {date}, closest date = {closest_date}') # later
            if not closest_date:  # if this is already out of bounds
                # remove all rows from this point on from micro_x:
                micro_x.drop(micro_x.loc[date:].index, inplace=True)
                break
            else:
                matching_dates.append(closest_date)

        # add all rows of desired dates:
        svi_y = pd.concat([svi_y, svi0.loc[matching_dates]])

        assert (
            svi_y.shape[0] == micro_x.shape[0]
        ), f"x and y for bio reactor {bio_reactor_i} not same length"
        return micro_x, svi_y

    def find_closest_date(self, bio_reactor_i: int, date0):
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
            print("out of bounds")  # later
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
    data = ml_prepare()
    data.plot_svi()
    x, y = data.create_x_y_delayed(days_delay=5)

