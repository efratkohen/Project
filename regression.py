from ml_prepare import ML_prepare

if __name__=='__main__':
    data = ML_prepare()
    # data.plot_svi()
    delay = 4
    x, y = data.create_x_y_delayed(days_delay=delay)