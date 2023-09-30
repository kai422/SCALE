from torchvision.models.resnet import Bottleneck, ResNet


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # self.input = input.clone()
        self.output = output.clone()

    def close(self):
        self.hook.remove()


class ResNet50(ResNet):
    def __init__(self,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 num_classes=1000):
        super(ResNet50, self).__init__(block=block,
                                       layers=layers,
                                       num_classes=num_classes)
        self.feature_size = 2048
        self._prerelu = SaveFeatures(self.layer4[2].bn3)

    def forward(self, x, return_feature=False, return_feature_list=False, return_prerelu_feature=False):
        if return_prerelu_feature:
            feature1 = self.relu(self.bn1(self.conv1(x)))
            feature1 = self.maxpool(feature1)
            feature2, feature2_prerelu = self.layer1([feature1])
            feature3, feature3_prerelu = self.layer2([feature2])
            feature4, feature4_prerelu = self.layer3([feature3])
            feature5, feature5_prerelu = self.layer4([feature4])
            feature5 = self.avgpool(feature5)
            feature = feature5.view(feature5.size(0), -1)
            return feature,  self.avgpool(feature5_prerelu).view(feature5.size(0), -1)

        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.fc(feature)

        feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x, threshold):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)

        return logits_cls

    def intermediate_forward(self, x, layer_index):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        if layer_index == 1:
            return out

        out = self.layer2(out)
        if layer_index == 2:
            return out

        out = self.layer3(out)
        if layer_index == 3:
            return out

        out = self.layer4(out)
        if layer_index == 4:
            return out

        raise ValueError

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc
