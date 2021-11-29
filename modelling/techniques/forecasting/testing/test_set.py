import calendar, random
from datetime import datetime
import calendar
import random
from datetime import datetime

random.seed(0)
from datetime import timedelta


def generate_testset_baseline(start_date, end_date, interval, output_path):
    file_to_write = open(output_path + start_date + "_" + end_date + "_" + str(interval) + "_baseline.txt", 'w')
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    for dt in daterange(start, end):
        file_to_write.write(dt.strftime("%Y-%m-%d") + "\n")
    file_to_write.close()

def daterange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(n)

# All crypto series are observed until 31th of January 2019 (2019-12-31)
# It takes the starting and ending date and "number_samples" entries choosed randomly

def generate_testset(start_date, end_date, output_path):
    file_to_write = open(output_path + start_date + "_" + end_date + ".txt", 'w')
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # get the number of months between two dates
    num_months = (end.year - start.year) * 12 + end.month - start.month
    num_months = num_months + 1

    test_set = set()
    for i in range(0, num_months):
        test_set_specific = set()
        while len(test_set_specific) < 3:
            random_day = randomdate(start.year, start.month)
            # if the last random day generated is upper than the last available day, by default it will be set up to the last available day
            # Example:
            # end date: 18-01-2019, thus last available day is: 18.
            # random date generated: 19-01-2019
            # thus, the random date day will be 18 instead of 19.
            if i + 1 == num_months and random_day.day > end.day:
                random_day = random_day.replace(day=end.day)

            # se non ci sono 5 date, niente da fare.
            test_set_specific.add(random_day)
        test_set = test_set | test_set_specific

        # update the new start date and end date (it generates a date per months!)
        new_year = start.year
        new_month = start.month + 1  # find a new date in the next month
        if (new_month == 13):
            new_month = 1
            new_year = start.year + 1  # find a new date in the next year
        start = start.replace(year=new_year, month=new_month)

    test_set = list(test_set)
    test_set.sort()
    # test_set=set(test_set)
    for date in test_set:
        # adding the random day to the list
        file_to_write.write(str(date) + "\n")
    # file_to_write.write(str(test_set))
    file_to_write.close()
    return


def randomdate(year, month):
    # The itermonthdates() method returns an iterator for the month (1-12) in the year.
    # This iterator will return all days (as datetime.date objects) for the month and all days before the start of the month
    # or after the end of the month that are required to get a complete week.
    dates = calendar.Calendar().itermonthdates(year, month)
    # get only the dates of our month of interest
    dates_of_the_month = [date for date in dates if date.month == month]
    return random.choice(dates_of_the_month)


def get_testset(path_file):
    test_set=[]
    with open(path_file) as td:
        test_dates = td.readlines()

    for test_date in test_dates:
        test_set.append(test_date.replace("\n",""))

    return test_set