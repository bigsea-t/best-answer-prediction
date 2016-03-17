# This script is for reformatting the XML archieve of stackoverflow into JSON

import regex
import html
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
K = 50000
pattern = regex.compile("\s+<row (?:(.*?)=\"(.*?)\" )+\/>\s+")
qs = dict()
rqs = []
other = []
discarded = set()
count = 0
rqslock = threading.Lock()
def checkqs():
    pass
def dumprqs():
    global count
    global rqs
    rqslock.acquire()
    f = open("E:\Posts{}.dat".format(count), "w")
    json.dump(rqs, f)
    count += 1
    rqs = []
    f.close()
    rqslock.release()
def parseOneLine(s):
    global qs
    global rqs
    global other
    global discarded
    try:
        res = dict(zip(*(pattern.match(s).captures(1,2))))
    except Exception as e:
        print(s)
        print(e)
        raise e
    res['Body'] = html.unescape(res['Body'])
    if res['PostTypeId']=='1':
        if int(res['AnswerCount'])<0:
            discarded.add(res['Id'])
            return
        if res['Id'] in qs:
            qs[res['Id']].update(res)
        else:
            res['Children'] = []
            checkqs()
            qs[res['Id']] = res
        if int(res['AnswerCount']) == len(qs[res['Id']]['Children']):
            rqs.append(qs.pop(res['Id']))
            if len(rqs)==K:
                dumprqs()
    elif res['PostTypeId']=='2':
        if res['ParentId'] in discarded:
            return
        if res['ParentId'] in qs:
            qs[res['ParentId']]['Children'].append(res)
        else:
            qs[res['ParentId']] = {'Children':[res]}
        if "AnswerCount" in qs[res['ParentId']] and int(qs[res['ParentId']]['AnswerCount']) == len(qs[res['ParentId']]['Children']):
            rqs.append(qs.pop(res['ParentId']))
            if len(rqs)==K:
                dumprqs()
    else:
        other.append(res)
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Usage: python reformat_xml_to_json.py <XML data>')
    with ThreadPoolExecutor(max_workers=100) as executor:
        for i in open(sys.argv[1], encoding="utf-8"):
            if "<?xml" in i or i=="<posts>\n":
                continue
            a = executor.submit(parseOneLine, i)
