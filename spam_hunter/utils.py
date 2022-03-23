import time


class Clock:
    """The clock object stores start times, allowing the tracking of compute time for multiple processes."""

    def __init__(self):
        """Constructor method"""
        self.start_times = {"origin": time.time()}

    def elapsed_time(self):
        """Compute the elapsed time since Clock was initiated
        :return: Number of seconds elapsed since Clock object was instantiated
        :rtype: float
        """
        return round(time.time() - self.start_times["origin"], 5)

    def start(self, clock_id):
        """Start the clock for specified clock_id by recording current time in dictionary
        :param clock_id: Unique ID for each timing process, which will be tracked separately from other IDs
        :type clock_id: any
        """
        self.start_times[clock_id] = time.time()

    def stop(self, clock_id):
        """Stop the clock for specified clock_id by calculating elapsed time and printing
        :param clock_id: Unique ID for each timing process, which will be tracked separately from other IDs
        :type clock_id: any
        :return: Elapsed time since start time for clock_id (info message if clock_id has not been started)
        :rtype: float (str if clock_id has not been started)
        """
        start_time = self.start_times.get(clock_id)
        if start_time is None:
            return f"Clock {clock_id} not started"
        else:
            return round(time.time() - start_time, 5)

c = Clock()
