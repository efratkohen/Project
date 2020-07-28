from ml_prepare import Ml_prepare

if __name__=='__main__':
    data = Ml_prepare()
    data.plot_svi()
    delay = 4
    # x, y = data.create_x_y_delayed(days_delay=delay)