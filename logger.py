#logger.py

import time

class Logger():
    """Class: Logger

        Description: Util class for logging. All test results should be logged using this class.

    """
    
    #Internal
    def __init__(self, model_name):
        """Function: __init__

            Description:
                Initiate new logger class. logging path of this logger is set to "logs/" + model_name + ".log"

            Args:
                model_name (str): log name. If a log that have same name exists, logger will append logs after it.

            Attributes:
                logger (file): a file object pointing log.

            Returns:
                None
        """
        self.logger = open("logs/" + model_name + ".log", "a+")
        print("Logger has been created.")

    def __del__(self):
        self.logger.close()

    #API
    def log(self, text, write_time = False):
        """Function: log

            Description:
                Write given data to log. If write_time is set True, calender time will be recorded too.

            Args:
                text (str): text.
                write_time (bool): write down current calender time to log if set True, Default is False.

            Returns:
                None but log is updated.
        """
        if write_time:
            self.logger.write(str(time.ctime(time.time())) + "\n")
        self.logger.write(text + "\n")

