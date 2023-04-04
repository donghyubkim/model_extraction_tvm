def jump(c,ind,skip):
    
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
               'q','r','s','t','u','v','w','x','y','z']
    current = alphabet.index(c)
    
    while ind:
        current +=1
        if alphabet[current % 26] in skip:
            continue
        else:
            ind-=1
    
    return alphabet[current]
            
            

def solution(s, skip, index):
    
    s = [i for i in s]
    
    for i in range(len(s)):
        s[i]=jump(s[i],index,skip)
    
    answer = "".join(s)
        
    return answer
solution('aukks','wbqd',5)