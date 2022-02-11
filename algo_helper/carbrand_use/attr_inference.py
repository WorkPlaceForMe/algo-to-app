# encoding: utf-8

import json

import numpy as np
import torch
from PIL import Image


class VehicleAttrRecog:
    def __init__(self, model_weights, class_file):
        self.pt_sess = self.build_pt_runtime(model_weights)

        # pre-process
        self.size_test = 256

        # post-process
        self.idx2class = self.build_idx2class(class_file)

    @classmethod
    def build_idx2class(cls, class_file):
        with open(class_file, 'r') as f:
            idx2class = json.load(f)
        return idx2class

    @classmethod
    def build_pt_runtime(cls, model_weights):
        pt_sess = torch.jit.load(model_weights)
        pt_sess.eval()
        return pt_sess

    def preprocess(self, images):
        inp_tensors = []
        for image in images:
            res_img = image.resize((self.size_test, self.size_test), Image.CUBIC)
            inp_tensor = torch.from_numpy(np.array(res_img, np.float32, copy=False))
            inp_tensor = inp_tensor.transpose(0, 1).transpose(0, 2).contiguous()
            inp_tensors.append(inp_tensor)
        inp_tensors = torch.stack(inp_tensors)
        return inp_tensors.cuda()

    def postprocess(self, pred_logits):
        pred_res = []
        start = 0
        for attr_dict in self.idx2class:
            end = start + len(attr_dict)
            attr_logits = torch.softmax(pred_logits[:, start:end], dim=1)
            pred_scores, pred_indices = torch.max(attr_logits, dim=1)

            attr_res = []
            for i in range(len(pred_scores)):
                pred_score = pred_scores[i].item()
                label_index = str(pred_indices[i].item())
                pred_class = attr_dict[label_index]
                attr_res.append([pred_class, pred_score])

            start = end
            pred_res.append(attr_res)

        return pred_res

    def __call__(self, images):
        """
        Get the cropped image, and predict the character.

        Args:
            images (list): List of PIL.Image

        Returns:
            List of predict character.
        """
        inp_tensors = self.preprocess(images)
        pred_logits = self.pt_sess(inp_tensors)
        pred_res = self.postprocess(pred_logits)
        return pred_res


if __name__ == '__main__':
    AttrRecogEngine = VehicleAttrRecog("pt_exports/r50_attr.pt", "pt_exports/idx2attr.json")

    from PIL import Image

    img = [Image.open("pt_exports/Volkswagen.jpg"), ]
    pred_res = AttrRecogEngine(img)
    for pred in pred_res:
        for res in pred:
            print(f"vehicle predicted attribute class is {res[0]}, probability is {res[1]:.3f}")
