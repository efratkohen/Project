# class ml_prepare, test creating delayed x y    
    dif = timedelta(days = delay)
    for i in range(1,5):
        xi = x.loc[f'{i}']
        yi = y.loc[f'{i}']
        assert xi.shape[0]==yi.shape[0]
        for row_i in range(len(xi.index)):
            assert yi.index[row_i] == xi.index[row_i] + dif