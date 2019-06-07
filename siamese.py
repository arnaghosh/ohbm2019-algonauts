from Feature_Extract.alexnet import * # change name for different model

class SiameseModel(nn.Module):
    def __init__(self,layer):
        super(SiameseModel, self).__init__()
        self.model = AlexNet() #change name for different model
        self.layer_to_opt = layer 
        
    def forward(self, inp1, inp2):
        feats1_list = self.model(inp1);
        feats1 = feats1_list[self.layer_to_opt];
        feats2_list = self.model(inp2);
        feats2 = feats2_list[self.layer_to_opt];
        # print(torch.mean(feats1),torch.mean(feats2))
        # print(feats1.shape,feats2.shape)
        feats1 = feats1 - torch.mean(feats1,dim=0,keepdim=True)
        feats2 = feats2 - torch.mean(feats2,dim=0,keepdim=True)
        # print(torch.mean(feats1),torch.mean(feats2))
        # print(feats1.shape,feats2.shape)
        sim = torch.sum(feats1*feats2, dim=1)*torch.rsqrt(torch.sum(feats1*feats1,dim=1))*torch.rsqrt(torch.sum(feats2*feats2,dim=1))
        return 1-sim
        