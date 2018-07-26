import hashlib
import random
import urllib.request    
import json
import http.client

def translate(text):
    appid = '20151113000005349'
    secretKey = 'osubCEzlGjzvw8qdQc41'
    
    myurl = '/api/trans/vip/translate'
    q = text.replace(' ','|')
    fromLang = 'en'
    toLang = 'zh'
    salt = random.randint(32768, 65536)

    sign = appid+q+str(salt)+secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode("utf8"))
    sign = m1.hexdigest()
    myurl = myurl+'?appid='+appid+'&q='+q+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
    httpClient=None
    try:

        con = http.client.HTTPConnection('api.fanyi.baidu.com')

        con.request("GET", myurl,'',{})
        resu = con.getresponse()
        #print (resu)
        #print(resu.status)
        #print(resu.reason)
        #print(resu.info())
        html=resu.read()    
        #print (html)
        text = json.loads(html)
        #print (text)
        return text.get("trans_result")[0].get("dst")
    except Exception as e:
        print (e)
    finally:
        if httpClient:
            httpClient.close()

