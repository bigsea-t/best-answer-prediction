import re

code_pattern = re.compile("<code>([\s\S]*?)</code>")

def CountCodeLines(text):
    codes = code_pattern.findall(text)
    return sum([i.count('\n') for i in codes])

def CountCodeChars(text):
    codes = code_pattern.findall(text)
    return sum([len(i) for i in codes])

def RemoveCode(text):
    return code_pattern.replace(' ', text)
    
def SentenceStructure(sentence):
    pass

def CountChars(text):
    return len(text)