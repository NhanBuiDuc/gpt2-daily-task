from task import Task, TasksManager
import datetime
from datetime import datetime, timedelta, time

target_text = [
    "<sum>Prepare presentation for 18 this month<cate>Work<prio>3<diff>4<imp>4<status>0<exp_min>120<totd>2<spec_time>null<dow>null<day>18<month>null<no_date>3<no_week>null<no_month>0",
    "<sum>Finish coding assignment before tommorow<cate>Study<prio>2<diff>1<imp>1<status>0<exp_min>180<totd>2<spec_time>null<dow>null<day>null<month>null<no_date>2<no_week>null<no_month>null",
    "<sum>Buy groceries in Thursday<cate>Errands<prio>5<diff>5<imp>3<status>0<exp_min>60<totd>2<spec_time>null<dow>5<day>null<month>null<no_date>1<no_week>null<no_month>null"
]

# sắp xếp task theo priority
def sort_task(task_list):
    return sorted(task_list, key=lambda x: x.priority)


# Chọn task để làm trong một buổi (trong 5 buổi)
def task_filter(task_in_same_period : list, start_time, end_time):
    # sắp xếp task
    task_in_same_period = sort_task(task_in_same_period)

    # task được chọn để làm 
    remain_task = []  

    # task quá deadline và có khả năng làm được trong ngày khác
    filtered_task = []

    for task in task_in_same_period:
        task_time = task.expected_minutes
        if start_time + task_time > end_time:
            filtered_task += [task]
            continue
        remain_task += [task]
        start_time += task_time

    return remain_task, filtered_task

def group_task(task_list):
    task_list = sorted(task_list, key=lambda x : x.remain_time)

    # Define the threshold (3 days)
    threshold_days = 3
    # Initialize lists for normal and emergency dates
    normal_tasks = []
    emergency_tasks = []

    # Get the current date and time
    #current_date_time = datetime.now()


    # Iterate through the list of tasks
    for task in task_list:
        # Get the total number of days from the timedelta
        remaining_days = task.remain_time.days

        # Determine whether it's normal or emergency task
        if remaining_days > threshold_days:
            normal_tasks.append(task)
        else:
            emergency_tasks.append(task)

    return normal_tasks, emergency_tasks

def to_do_list_in_day(task_list):
    tasks = [[], [], [], [], []]
    remain_task = [[], [], [],
                   
                    [], []]
    filtered_task = [[], [], [], [], []]

    for task in task_list:
        tasks[task.time_of_day] += [task]
    
    start_time = [0, 240, 480, 720, 960, 1200] 
    end_time   = [240, 480, 720, 960, 1200, 1440]
    for i in range(0, 5):
        remain_task[i], filtered_task[i] = task_filter(tasks[i], start_time[i], end_time[i])

    return remain_task, filtered_task

def select_task_for_a_day(tasks):
    normal_tasks, emergency_tasks = group_task(tasks)
    remain_task_1, filtered_task_1 = to_do_list_in_day(emergency_tasks)
    remain_task_2, filtered_task_2 = to_do_list_in_day(normal_tasks)

    remain_task = [[], [], [], [], []]
    filtered_task = [[], [], [], [], []]


    for i in range(0, 5):
        remain_task[i] = remain_task_1[i] + remain_task_2[i]
        filtered_task[i] = filtered_task_1[i] + filtered_task_2[i] 
    
    ret_task = []
    for tasks in filtered_task:
        for task in tasks:
            ret_task += [task]

    return remain_task, ret_task

def to_hour(minutes):
    hours = minutes // 60  # Số giờ là phần nguyên của phép chia
    remainder_minutes = minutes % 60  # Số phút dư
    time_object = time(hours, remainder_minutes)
    return time_object

def process(tasks):
    ret_dict = {}
    start_time = [0, 240, 480, 720, 960, 1200] 
    for i in range(0, 5):
        st = start_time[i]
        for task in tasks[i]:
            ret_dict[str(to_hour(st))] = task.id
            st += task.expected_minutes

    return ret_dict

import sys
if __name__ == "__main__":
    tasks = TasksManager()
    for text in target_text:
        tasks.add_task(text)
    
    final_dict = {}
    current_date = datetime.now()
    delta = 0
    while len(tasks) != 0:
        remain_task, tasks = select_task_for_a_day(tasks)
        task_dict = process(remain_task)
        final_dict[str((current_date + timedelta(days=delta)))] = task_dict
        delta += 1
        
    print(final_dict)
        

