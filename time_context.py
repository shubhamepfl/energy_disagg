import numpy as np
import pandas as pd
from datetime import time

def time_context(data,appliance_name):
    col =["{}_{}".format(appliance_name,i) for i in xrange(2)]
    time_x=pd.DataFrame(index=data.index, columns=col)
    time_x["{}_0".format(appliance_name)] =(time_x.index.time>time(9)) & (time_x.index.time<=time(17))
    time_x["{}_1".format(appliance_name)] =(time_x.index.time>time(17)) | (time_x.index.time<=time(9))
    return pd.concat([data, time_x], axis=1)

