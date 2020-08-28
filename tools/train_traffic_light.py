from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import time
import torch

from deep_sort.cosine_metric_net import TrafficNet


class TrainDataset(Dataset):
    def __init__(self, path_list, label_list):
        super(TrainDataset, self).__init__()
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.path_list = path_list
        self.label_list = label_list
        assert len(path_list) == len(label_list), "Length of both lists should equal!"

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, item):
        img = Image.open(self.path_list[item]).convert("RGB")
        img = self.transforms(img)
        return img, self.label_list[item]


class TestDataset(TrainDataset):
    def __init__(self, path_list, label_list):
        super(TestDataset, self).__init__(path_list, label_list)
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


if __name__ == '__main__':
    max_epoch = 100

    img_dataset = datasets.ImageFolder(root='./traffic-light')
    X, Y = zip(*img_dataset.imgs)  # img_path and labels
    print(img_dataset.class_to_idx)  # class -> index mapping

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=233,
                                                        stratify=Y)
    train_dataset = TrainDataset(X_train, Y_train)
    test_dataset = TestDataset(X_test, Y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=12, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=12, drop_last=False)

    criterion = nn.CrossEntropyLoss()  # for CosineMetricLearning`
    model = TrafficNet(num_classes=len(img_dataset.class_to_idx)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    for epoch in range(1, max_epoch + 1):
        model.train()
        training_loss = 0.0
        correct, total = 0, 0
        st_time = time.time()
        for iter_num, batched_data in enumerate(train_dataloader):
            img, img_labels = batched_data
            img, img_labels = Variable(img).cuda(), Variable(img_labels).cuda()

            outputs = model(img)
            loss = criterion(outputs, img_labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # accumurating
            training_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(img_labels).sum().item()
            total += img_labels.size(0)

            if (iter_num + 1) % 1 == 0:
                print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                    100. * (iter_num + 1) / len(train_dataloader), time.time() - st_time, training_loss / 10, correct,
                    total,
                    100. * correct / total
                ))
                st_time = time.time()

        if epoch % 3 == 0:
            model.eval()
            test_loss = 0.
            correct, total = 0, 0
            st_time = time.time()
            for test_iter_num, batched_data in enumerate(test_dataloader):
                img, img_labels = batched_data
                img, img_labels = Variable(img).cuda(), Variable(img_labels).cuda()

                outputs = model(img)
                loss = criterion(outputs, img_labels)
                test_loss += loss.item()
                correct += outputs.max(dim=1)[1].eq(img_labels).sum().item()
                total += img_labels.size(0)
            print('Testing...')
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (test_iter_num + 1) / len(test_dataloader), time.time() - st_time, test_loss / len(test_dataloader), correct, total,
                100. * correct / total
            ))
    save_ckpt = {
        'model_dict': model.state_dict(),
        'type': img_dataset.class_to_idx
    }
    torch.save(save_ckpt, './trafficnet.pt')