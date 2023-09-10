import datetime

# Parse the time string "07:00:00"
time_str = "07:00:00"
time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S").time()

# Combine the current date with the parsed time to create a datetime object
combined_datetime = datetime.datetime.combine(datetime.datetime.now(), time_obj)
print(combined_datetime)