import torch
import torch.nn as nn
import torch.nn.functional as F
from AttributeClassifier import AttributeClassifier

class AttributeClassifierInference(nn.Module):
    def __init__(self, checkpoint_filename=None, attribute_classifier=None, device='cuda'):
        super(AttributeClassifierInference, self).__init__()
        if attribute_classifier:
            self.attribute_classifier = attribute_classifier
        elif checkpoint_filename:
            self.attribute_classifier = AttributeClassifier(out_features=359, device=device)
            checkpoint = torch.load(checkpoint_filename)
            self.attribute_classifier.load_state_dict(checkpoint['model'])
            self.saved_opt = checkpoint['settings']
        else:
            assert "No model passed in or checkpoint filename given"

        self.attribute_classifier.eval()
        self.eval()

    def forward(self, input):
        probs = torch.sigmoid(self.attribute_classifier(input))
        return torch.round(probs)