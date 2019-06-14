import torch
import torch.nn as nn

class loss_function(nn.Module):
    def __init__(self, args):
        super(loss_function, self).__init__()
        self.lossfn = nn.MSELoss(reduction='sum')
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
    
    def lossfn_two_var(self, target1, target2, num_px = None):
        if num_px is None:
            return torch.sum(torch.pow((target1 - target2),2))
        else:
            return torch.sum(torch.pow((target1 - target2),2) / num_px)

    def forward(self,output,GT_src_mask, GT_tgt_mask):
        eps = 1
        
        b, _, h, w = GT_src_mask.size()
        src_num_fgnd = GT_src_mask.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True) + eps
        tgt_num_fgnd = GT_tgt_mask.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True) + eps

        L1 = self.lossfn(output['est_src_mask'], GT_src_mask) / (h*w) + self.lossfn(output['est_tgt_mask'],GT_tgt_mask) / (h*w) # mask consistency
        L2 = self.lossfn_two_var(output['flow_S2T'], output['warped_flow_S2T'], src_num_fgnd)\
           + self.lossfn_two_var(output['flow_T2S'], output['warped_flow_T2S'], tgt_num_fgnd) # flow consistency
        L3 = torch.sum(output['smoothness_S2T'] / src_num_fgnd) + torch.sum(output['smoothness_T2S'] / tgt_num_fgnd) # smoothness
        
        return (self.lambda1*L1 + self.lambda2*L2 + self.lambda3*L3) / GT_src_mask.size(0), \
                L1*self.lambda1 / GT_src_mask.size(0),\
                L2*self.lambda2 / GT_src_mask.size(0),\
                L3*self.lambda3 / GT_src_mask.size(0)
