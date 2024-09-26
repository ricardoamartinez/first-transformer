from collections import Counter
import os
import time
import pprint
import re

class Tokenize:
    def bp_encoding(self,text):
        sorted_set = sorted(set(text))
        self.mapping = {i:ch for i,ch in enumerate(sorted_set)}
        temp = {ch:ch for i,ch in self.mapping.items()}
        idx = len(sorted_set)
        self.len_orig = len(text)
        self.encoded_text = text
        self.max_len = 0
        for i in range(1000,100000):
            if len(self.encoded_text)<=1 or self.len_orig/len(self.encoded_text)>4:
                break
            
            pairs = Counter(self.encoded_text[j:j+2] for j in range(len(self.encoded_text) - 1))

            most_common_pair, occurence = pairs.most_common(1)[0]

            if occurence==1:
                break

            new_pair = str(temp.get(most_common_pair[0])+str(temp.get(most_common_pair[1])))
            self.max_len = max(self.max_len,len(new_pair))
            temp[chr(i)] = new_pair
            self.mapping[idx] = new_pair
            self.encoded_text = self.encoded_text.replace(most_common_pair,chr(i))
            idx+=1
        print(self.len_orig/len(self.encoded_text))
        self.vocab_size = idx
        self.encode_map = {ch:i for i,ch in self.mapping.items()}


    def encode(self,text):
        encoded = []
        i = 0
        while i<=(len(text)-self.max_len):
            for j in range(i+self.max_len,i,-1):
                if text[i:j] in self.encode_map:
                    encoded.append(self.encode_map[text[i:j]])
                    i=j-1
                    break
            i+=1
        while i<len(text):
            for j in range(len(text),i,-1):
                if text[i:j] in self.encode_map:
                    encoded.append(self.encode_map[text[i:j]])
                    i=j-1
                    break
            i+=1
        return encoded

    def decode(self,encoded):
        return ''.join([self.mapping.get(x) for x in encoded])

# text = ''.join(open('test.txt','r').readlines())
# tokenizer = Tokenize()
# tokenizer.bp_encoding(text)
# encoded = tokenizer.encode(text)
# print(encoded)
# print(tokenizer.decode(encoded))
# while(True):
#     inp = input()
#     if inp=='quit':
#         break
#     out = tokenizer.encode(inp)
#     print(f"Encoded: {len(out)}, Original {len(inp)}")
#     print("Decoded:",tokenizer.decode(out))