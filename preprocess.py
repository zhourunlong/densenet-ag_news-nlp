import csv

LEN = 1012
LST = {' ': 0, '!': 1, '"': 2, '#': 3, '$': 4, '&': 5, "'": 6, '(': 7, ')': 8, '*': 9, ',': 10, '-': 11, '.': 12,
       '/': 13, '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23, ':': 24,
       ';': 25, '=': 26, '?': 27, '\\': 28, '_': 29}

def trans(s):
    n = len(s)
    ret = []
    for i in range(n):
        if (s[i] in LST.keys()):
            ret.append(LST[s[i]])
        elif ('A' <= s[i] and s[i] <= 'Z'):
            ret.append(ord(s[i]) - ord('A') + 40)
        else:
            ret.append(ord(s[i]) - ord('a') + 40)
    m = LEN - len(ret)
    for i in range(m):
        ret.append(0)
    return ret

def genconfig():
    tr_dt, tr_lb, te_dt, te_lb = [], [], [], []
    with open('./ag_news_csv/train.csv', mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
        for line in reader:
            s = "{} {}".format(line['title'], line['description'])
            l = int(line['label']) - 1
            tr_dt.append(trans(s))
            tr_lb.append(l)
    with open('./ag_news_csv/test.csv', mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
        for line in reader:
            s = "{} {}".format(line['title'], line['description'])
            l = int(line['label']) - 1
            te_dt.append(trans(s))
            te_lb.append(l)
    return tr_dt, tr_lb, te_dt, te_lb

if __name__ == '__main__':
    tr_dt, tr_lb, te_dt, te_lb = genconfig()
    print(tr_dt[: 1])
    print(tr_lb[: 1])
    print(te_dt[: 1])
    print(te_lb[: 1])