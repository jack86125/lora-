def second2time(seconds: int):
    """
    将秒转换成时分秒。

    Args:
        seconds (int): _description_
    """
    m, s = divmod(seconds, 60)
    print(f'm-->{m}')
    print(f's-->{s}')
    h, m = divmod(m, 60)
    print(f'h-->{h}')
    print(f'm-->{m}')
    return "%02d:%02d:%02d" % (h, m, s)

seconds = 6100
a = second2time(seconds)
print(a)