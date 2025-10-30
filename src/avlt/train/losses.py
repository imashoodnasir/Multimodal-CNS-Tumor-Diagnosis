
import torch
import torch.nn as nn
import torch.nn.functional as F

class Losses:
    def __init__(self, w_align=1.0, w_sd=0.5):
        self.w_align = w_align
        self.w_sd = w_sd
        self.ce = nn.CrossEntropyLoss()

    def align(self, f_v, f_t):
        f_vn = F.normalize(f_v, dim=1)
        f_tn = F.normalize(f_t, dim=1)
        return 1.0 - (f_vn * f_tn).sum(dim=1).mean()

    def distill(self, p_teacher, p_student):
        log_p_s = F.log_softmax(p_student, dim=1)
        p_t = F.softmax(p_teacher.detach(), dim=1)
        return F.kl_div(log_p_s, p_t, reduction='batchmean')

    def classification(self, logits, y):
        return self.ce(logits, y)

    def total(self, logits_s, logits_t, y, f_v, f_t):
        l_cls = self.classification(logits_s, y)
        l_align = self.align(f_v, f_t)
        l_sd = self.distill(logits_t, logits_s)
        total = l_cls + self.w_align*l_align + self.w_sd*l_sd
        return total, {"cls": l_cls.item(), "align": l_align.item(), "sd": l_sd.item()}
