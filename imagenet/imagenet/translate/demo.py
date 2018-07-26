#/usr/bin/env python
#coding=utf8
 
import hashlib
import random
import urllib.request    
import json
import http.client

def translate(text):
    appid = '20151113000005349'
    secretKey = 'osubCEzlGjzvw8qdQc41'
    
    myurl = '/api/trans/vip/translate'
    q = text.replace(", ","/n").replace(" ","")
    fromLang = 'en'
    toLang = 'zh'
    salt = random.randint(32768, 65536)

    sign = appid+q+str(salt)+secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode("utf8"))
    sign = m1.hexdigest()
    print (q)
    myurl = myurl+'?appid='+appid+'&q='+q+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
    print (myurl)
    httpClient=None
    try:
        con = http.client.HTTPConnection('api.fanyi.baidu.com')
        con.request("GET", myurl,'',{})
        resu = con.getresponse()
        html=resu.read()
        print (html)
        text = json.loads(html)
        print (text)
        return text.get("trans_result")[0].get("dst")
    except Exception as e:
        print (e)
    finally:
        if httpClient:
            httpClient.close()


filename = 'imagenet_synset_to_human_label_map.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行读取数据
        if not lines:
            break
        n_tmp, E_tmp = [str(i) for i in lines.split("	")] # 将整行数据分割处理
        print (E_tmp)
        C_tmp =translate(E_tmp) # 读取的后面的值进行翻译
        f = open('imagenet_synset_to_human_label_map_cn.txt','a')
        f.write(str(n_tmp)+"	"+str(C_tmp).replace(',',', ')+"\n")
        f.close()
