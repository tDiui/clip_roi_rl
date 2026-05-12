import torch
def calculate_iou(pred_segment, gt_segment):
    """
    pred_segment, gt_segment: tensor [2]
    """

    s_p, e_p = pred_segment[0], pred_segment[1]
    s_g, e_g = gt_segment[0], gt_segment[1]

    inter_start = torch.maximum(s_p, s_g)
    inter_end = torch.minimum(e_p, e_g)

    intersection = torch.clamp(inter_end - inter_start, min=0)

    union = (e_p - s_p) + (e_g - s_g) - intersection

    iou = intersection / (union + 1e-6)

    return iou.item()   # vẫn trả float nếu bạn cần