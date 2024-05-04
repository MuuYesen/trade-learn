

class SimpleWeight:

    def __init__(self):
        pass

    @staticmethod
    def weight_by_equal(stock_list):
        return [(1 - 0.05) / len(stock_list)]*len(stock_list)

    @staticmethod
    def weight_by_industry():
        pass
