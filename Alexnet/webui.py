import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型结构
class CustomAlexNet(torch.nn.Module):
    def __init__(self):
        super(CustomAlexNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 56 * 56, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 加载模型
model_path = 'models/model_epoch_20.pth'
assert os.path.exists(model_path), f"模型文件未找到：{model_path}"
model = CustomAlexNet().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义预测函数
def predict(image):
    image = transform(image).unsqueeze(0).to(device)  # 预处理并移动到 GPU
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return "狗" if predicted.item() == 1 else "猫"

# 使用 Gradio 创建 Web 界面
with gr.Blocks() as demo:
    gr.Markdown("# 猫狗分类器")
    gr.Markdown("上传一张图片，我会告诉你它是猫还是狗！")
    image_input = gr.Image(type="pil")
    label_output = gr.Label()
    classify_button = gr.Button("分类")
    classify_button.click(predict, inputs=[image_input], outputs=[label_output])

# 启动 Web 界面
demo.launch(share=True)
