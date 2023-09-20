import datetime

# Parse the time string "07:00:00"
time_str = "07:00:00"
time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S").time()

# Combine the current date with the parsed time to create a datetime object
combined_datetime = datetime.datetime.combine(datetime.datetime.now(), time_obj)
print(combined_datetime)

self.custom_tokens = [

    # Level 1: Không bị chứa bởi tag nào, chỉ được chứa bởi <bos> và <eos>
    # gồm có 2 phần bắt đầu và kết thúc, chứa nhiều hơn 1 token đôi
    # token đôi: task có bắt đầu và kết thúc
    "<startofprompt>", "<endofpromt>",  # Bắt đầu và kết thúc của đầu vào
    # Bắt đầu và kết thúc của lệnh trong đầu ra
    "<startofcommand>", "<endofcommand>",

    # Level 2: Có thể bị chứa bởi tag level 1
    # gồm có 2 phần bắt đầu và kết thúc, chứa nhiều hơn 1 token đôi

    "<startoftask>", "<endoftask>",  # Bắt đầu và kết thúc của một công việc


    # Level 2: Bị chứa bởi tag lvl 1
    # gồm có 2 phần bắt đầu và kết thúc, không chứa token đôi

    # 2.1. Các attribute sử dụng chung cho nhiều các tag cấp cao hơn
    "<botl>", "<eotl>",  # begining of task list và end of task list, chứa

    # Level 3: Bị chứa bởi tag lvl 2, không chứa token đặc biệt, chỉ chứa token thường
    "<taskid>", "<assigned_time>",  # Những biến không được quyết đinh bởi đầu vào
    "<time>",
    "<command_list>",
    # Các action bao gồm thêm, xóa, cập nhật, swap, query, remind
    "<action>", "<issue_list>", "<target_list>",

    # Begin of existed list and End of existed list, là các công việc đã tồn tại
    "<boel>", "<eoel>",
    # Các tính chất của một công việc
    '<sum>', '<cate>', '<prio>', '<diff>', '<imp>', '<status>', '<exp_min>',
    '<totd>', '<spec_time>', '<dow>', '<day>', '<month>', '<no_date>', '<no_week>', '<no_month>'
]
