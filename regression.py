from ml_prepare import ml_prepare

if __name__=='__main__':
    data = ml_prepare()
    data.plot_svi()
    delay = 4
    # x, y = data.create_x_y_delayed(days_delay=delay)