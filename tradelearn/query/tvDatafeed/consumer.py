import threading, queue, traceback

class Consumer(threading.Thread):
    '''
    Seis data consumer and processor
    
    This object contains reference to Seis and callback function
    which will be called when new data bar becomes available for
    that Seis. Data reception and calling callback function is 
    done in a separate thread which the user must start by calling
    start() method.
    
    Parameters
    ----------
    seis : Seis
        Consumer receives data bar from this Seis
    callback : func
        reference to a function to be called when new data available,
        function protoype must be func_name(seis, data)
    
    Methods
    -------
    put(data)
        Put new data into buffer to be processed
    del_consumer()
        Shutdown the callback thread and remove from Seis
    start()
        start data processing and callback thread
    stop()
        Stop the data processing and callback thread
    '''
    def __init__(self, seis, callback):
        super().__init__()

        self._buffer=queue.Queue()               
        self.seis=seis
        self.callback=callback
        self.name=self.callback.__name__+"_"+self.seis.symbol+"_"+seis.exchange+"_"+seis.interval.value
    
    def __repr__(self):
        return f'Consumer({repr(self.seis)},{self.callback.__name__})'
    
    def __str__(self):
        return f'{repr(self.seis)},callback={self.callback.__name__}'
    
    def run(self):
        # callback thread tasks
        while True:
            data=self._buffer.get()
            if data is None:
                break

            try: # in case user provided function throws an exception
                self.callback(self.seis, data)
            except Exception as e: # remove the consumer from Seis and close down gracefully
                self.del_consumer()
                self.seis=None # delete references
                self.callback=None
                self._buffer=None
                raise e from None
        
        self.seis=None # delete references
        self.callback=None
        self._buffer=None
    
    def put(self, data):
        '''
        Put new data into buffer to be processed
        
        Parameters
        ----------
        data : pandas.DataFrame
            contains single bar data retrieved from TradingView
        '''
        self._buffer.put(data)
    
    def del_consumer(self, timeout=-1):
        '''
        Stop the callback thread and remove from Seis
        
        Parameters
        ----------
        timeout : int, optional
            maximum time to wait in seconds for return, default
            is -1 (blocking)
        
        Returns
        -------
        boolean
            True if successful, False if timed out.
        '''
        return self.seis.del_consumer(self, timeout)
    
    def stop(self):
        '''
        Stop the data processing and callback thread
        '''
        self._buffer.put(None)
        