import datetime

minutes = 240
hours = minutes // 60
remainder_minutes = minutes % 60

time_object = datetime.time(hours, remainder_minutes)