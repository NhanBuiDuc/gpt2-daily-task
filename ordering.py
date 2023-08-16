from task import Task, Tasks

target_text = [
    "<sum>Prepare presentation for 18 this month<cate>Work<prio>3<diff>4<imp>4<status>0<exp_min>120<totd>2<spec_time>null<dow>null<day>18<month>null<no_date>null<no_week>null<no_month>0",
    "<sum>Finish coding assignment before tommorow<cate>Study<prio>2<diff>1<imp>1<status>0<exp_min>180<totd>4<spec_time>null<dow>null<day>null<month>null<no_date>1<no_week>null<no_month>null",
    "<sum>Buy groceries in Thursday<cate>Errands<prio>5<diff>5<imp>3<status>0<exp_min>60<totd>3<spec_time>null<dow>5<day>null<month>null<no_date>null<no_week>null<no_month>null"
]

tasks = Tasks(target_texts=target_text)
print(tasks)
