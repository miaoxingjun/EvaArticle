import os,torch

from pro.allyears.basicfunc import *
from pro.deeplearn.trainmodel.utils.vocab import *
from pro.deeplearn.trainmodel.utils.ttspl import *


class Config(object):

    model_dict =  {
        1: 'LSTM',
        2: 'RNN',
        3: 'GRU',
        4: 'FastText',
    }

    optimdict = {
        1: 'Adam',
        2: 'SGD',
        3: 'Adagrad',
        4: 'RMSprop',
        5: 'AdamW',
        6: 'Adamax',
        7: 'ASGD',
        8: 'LBFGS'
    }

    costdict = {
        1: "torch.nn.MSELoss(reduction='mean').to(config.device)",
        2: "torch.nn.CrossEntropyLoss().to(config.device)",
        3: "torch.nn.BCEWithLogitsLoss().to(config.device)",
        4: "torch.nn.NLLLoss().to(config.device)",
        5: "torch.nn.L1Loss().to(config.device)"
    }

    datatypedict = {
        1:'word',
        2:'number',
        3:'wordprd',
        4:'pic',
    }

    traintypedict = {
        1:'ys',
        2:'sex',
        3:'god'
    }

    costname = {1: 'MSE', 2: '交叉熵', 3: '二分类', 4: 'NLL', 5: 'MAE'}

    def __init__(self):
        self.run = False
        self.device = torch.device('cuda:0')
        """
        数据处理部分
        """
        trainname = 'weibo'
        traintype = 'single'
        datatype = 1                               # 文本分类为1，数据时序预测为2，文本生成为3, 数组（图片）分类为4
        self.datatype = self.datatypedict[datatype]
        self.is_padsize = 40                  # 如果每句话长度相同选0，不同的话，取中值选1，取平均值选2，大于3则取这个值为位长
        self.num_workers = 0                 #  尽量设置成0
        self.jsonpath = r"F:\data/json"
        self.upsample = 120
        self.onehot = True                    # 是否独热
        if self.datatype in ['word','wordprd']:
            self.path = r"F:\data\weibo\weibo100.xlsx"  # 总的需要训练的excel文件
            self.features = ['content']  # 数据列，列表形式。如果有需要embedding的，放在features最前面
            self.label = ['single']  # 目标值
            # self.voca_dict = build_vocab(self.path,self.features[0])  # 形成一个字典{'字':编号}
            self.len_voca = 152064
            self.pad_size = calpadsize(self.path, self.features[0], self.is_padsize) if self.is_padsize < 3 else self.is_padsize
            self.endsymbol = '¶'
        if self.datatype == 'number':              # 回归问题
            self.path = r"F:\data\json\stock\day"  # 总的需要训练的文件或者文件夹
            self.datacut = (0,50)              # 如果总数据量过大，需要摘取部分数据，则设置这个
            self.startdata = 3000               # 如果单个数据量过大，需要摘取部分数据，则设置这个，代表了数据的起始位置
        if self.datatype == 'pic':
            self.upsample = 1                   # face_recognition的upsample
            self.picpath = r"F:/data/face"
            self.writedata = True              #  是否将数据写入json
            train_type = 1                      #  1:ys,2:sex,3:god
            self.traintype = self.traintypedict[train_type]
            self.dataname = fr'face_{trainname}_data_{self.upsample}_{self.traintype}.json'
        """
        模型参数
        """
        self.num_classes = 3 if datatype != 3 else self.len_voca   # 分类数，默认1是回归问题，非1是分类问题
        self.input_size = 100                     # seq_len，每句话输入的长度，或者数据的长度
        self.embed = 300                        # shape的最内部数据的长度
        if self.datatype == 'pic':
            self.embed = 128                        # 一定要关注face_recognition.face_encodings(img)中的shape是[128,1]还是[1,128]
        model_num = 1                        # 选择模型，1: 'LSTM',2: 'RNN',3: 'GRU', 4: 'FastText',
        self.bidirect = False                 #  是否使用bidirectional
        self.attention = False                      #  是否使用多头注意力。True是使用，False是不使用
        self.normal = True                       #  是否批量归一化
        self.hidden_size = 128                    #  LSTM单元隐藏层个数
        self.num_layers = 2                    #  LSTM层个数
        self.dropout = 0.5                      # 随机丢弃
        self.powerlst = False                   # 是否对目标赋予权重
        if self.run:
            self.dropout = 0
        self.batch_size = 64 if self.run == False else 1  # 批数据大小，如果显存溢出则调小，原则为2**n
        self.models = self.model_dict[model_num]
        self.mapping = True                   #  是否做映射，加relu函数，只用作model_num==4的时候
        if self.attention:
            self.attentionforce = False        #  选择是否对多头注意力进行加强
            self.nhead = 4                    #  必须能整除hidden_size
            self.head_layer = 1               #   如果使用transformerencoder，代表了transformer层数


        """
        训练及优化器参数
        """
        cost = 2 if self.num_classes != 1 else 1      #  1: 'MSE', 2: '交叉熵', 3: '二分类', 4: 'NLL', 5: 'MAE'
        optim = 5                             # 1: 'Adam',2: 'SGD',3: 'Adagrad',4: 'RMSprop',5: 'AdamW',6: 'Adamax',7: 'ASGD',8: 'LBFGS'
        self.weight_decay = 1e-1               # 正则化，不动
        self.epoch = 200                       #  步长
        self.change_lr = True                  #  是否learnrate可变
        self.learning_rate = 1e-2               #  学习率                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            #  学习率
        if self.change_lr:
            self.step_size = 5                 #  多少步变一次学习率
            self.gamma = 1-(self.step_size/self.epoch)
        self.optim = self.optimdict[optim]    #  和optim一起，选择优化器，这里不要动
        self.cost = self.costdict[cost]       #  和cost一起，选择损失函数，这里不要动
        self.lossname = self.costname[cost]    #  输出损失函数名称，不要动

        """
        模型打印及保存
        """
        if not self.run:
            self.debug = True                       #  调式模式，false为手动打断之后保存
            self.stopflag = False                   #  是否自动判断终止（不动这个）
            self.printtrainlog = 50                 #  打印结果的步数
            self.printflag = True                   #  是否在每outputepoch步保存模型文件
            self.savemodelnumber = 20               # 一共保存多少个模型文件
            self.outputepoch = self.epoch // self.savemodelnumber  # 每多少步保存模型文件
            self.typename = f'{trainname}-{traintype}'
            self.tm = numchangetime(time.time())
            self.name = f'{self.hidden_size}-{self.num_layers}---{self.num_classes}'
            dirname = f'{trainname}-{traintype}-upsample{self.upsample}' if self.upsample else f'{trainname}-{traintype}'
            ptpath = fr'save/{dirname}/{self.models}/{self.name}/{str(self.tm).split(" ")[0]}-{str(self.tm).split(" ")[1][:2]}-{str(self.tm).split(" ")[1][3:5]}'
            if not self.run:
                os.makedirs(ptpath,exist_ok=True)
                # print(f'模型文件保存路径：{ptpath}')
            self.savepath = fr'./{ptpath}/{self.epoch}-{self.learning_rate}' if not self.upsample else fr'./{ptpath}/{self.epoch}-{self.learning_rate}_{self.upsample}'


if __name__ == '__main__':

    import pro.deeplearn.trainmodel.wbmain