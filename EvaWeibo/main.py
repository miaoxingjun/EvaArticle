import os,torch,re
from config import Config
from transformers import AutoTokenizer
from pro.deeplearn.trainmodel.model.models import *
import functools

model = None
config = Config()
tokenizer = AutoTokenizer.from_pretrained(r'F:\models\deepseek-r1-14b')

class EvaWeibo:
    def __init__(self):
        pass
    def load_model(self):
        global model
        model = LSTMMultiheadAttention(config)
        param = torch.load(r"./ptfiles/evaarticle.pt")
        model.load_state_dict(param)
        model.eval()
    def getresult(self,contents):
        self.load_model()
        contents = tokenizer.encode(contents,return_tensors='pt').numpy().tolist()[0][1:config.input_size+1]
        if len(contents) < config.input_size:
            contents.extend([151643] * (config.input_size - len(contents)))
        contents = torch.tensor([contents])
        modelres = model(contents).to(config.device)
        res = F.softmax(modelres, dim=1)
        result = torch.max(res.data, 1).indices.cpu().numpy().tolist()[0]
        return result

if __name__ == '__main__':
    with open(r"C:\Users\Administrator\Desktop\ttt.txt",'r',encoding='utf-8') as f:
        word = f.read()
    a = EvaWeibo().getresult(word)
    print(a)
