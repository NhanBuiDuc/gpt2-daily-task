def find_and_replace(original_string, target_string, replace_string, replace_mode):
    assert(replace_mode in ["first", "all"]), r"Only support for 'first' mode or 'all' mode"
    max_len = 20000
    Z = [0 for i in range(max_len)] 

    Z_string = target_string + "#" + original_string
    n = len(Z_string)

    l = 0
    r = 0

    for i in range(1, n):
        if (r < i):
            l = r = i
            while (r < n and Z_string[r - l] == Z_string[r]):
                r += 1
            Z[i] = r - l
            r -= 1
        else:
            k = i - l
            if Z[k] < r - l + 1:
                Z[i] = Z[k]
            else:
                l = i
                while (Z_string[r - l] == Z_string[r]):
                    r += 1
                Z[i] =  r - l
                r -= 1
    
    len_target_string = len(target_string)
    idx_list = []

    for i in range(len_target_string + 1, n):
        if Z[i] >= len_target_string:
            idx_list += [i - len_target_string - 1]

    if len(idx_list) == 0:
        print("Not found !")
        return original_string
    
    if replace_mode == 'first':
        return original_string[:idx_list[0]] + replace_string + original_string[idx_list[0] + len_target_string : ]
    elif replace_mode == 'all':
        p = 0
        ret_string = ""
        while (p < len(original_string)):
            if (p in idx_list):
                p += len_target_string
                ret_string += replace_string
            else:
                ret_string += original_string[p]
                p += 1
        return ret_string
        

A = "manh oi manh oi manh a"
B = "manh"
C = find_and_replace(A, B, "t", replace_mode="all")
print(C)