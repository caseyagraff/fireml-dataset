import datetime as dt


def date_range(start_date, end_date, increment=dt.timedelta(hours=24)):
    cur_date = start_date

    while cur_date < end_date:
        yield cur_date
        cur_date += increment


def month_range(start, end):
    cur = dt.date(start.year, start.month, 1)
    while cur < end:
        yield cur
        year = cur.year + cur.month // 12
        month = (cur.month % 12) + 1
        cur = dt.date(year, month, 1)
