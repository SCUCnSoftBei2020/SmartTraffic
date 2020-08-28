import torch
from PIL import Image
from torchvision import transforms
from deep_sort.cosine_metric_net import TrafficNet

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def test(model, img: Image):
    img = test_transform(img)
    img = img.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(img)
        pred = output.max(dim=0)[1]
        pred = pred.squeeze()
    print(_type[int(pred)])


if __name__ == '__main__':
    ckpt = torch.load('./trafficnet.pt')
    type_mapping: dict = ckpt['type']  # {'green': 0}
    _type = {k: v for v, k in type_mapping.items()}

    model = TrafficNet(num_classes=len(type_mapping))
    model.load_state_dict(ckpt['model_dict'])

    img = Image.open('./red.png').convert("RGB")
    test(model, img)
