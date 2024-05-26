import torch
import time
from tqdm import tqdm
from net.jxtnet_autoencoderpure import MeshAutoencoder


start_time0 = time.time()
print('代码开始时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))  

epoch = 400
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device:{device}')

in_em = torch.tensor([90,330,2]).to(device) #入射波\theta \phi freq
planesur_face = torch.load('planesur_face.pt').to(device)
planesur_vert = torch.load('planesur_vert.pt').to(device)
planesur_faceedge = torch.load('face_edges.pt').to(device) #这个face_edges是图论边，不是物理边，那这个生成边的代码不用动。

autoencoder = MeshAutoencoder( #这里实例化，是进去跑了init
    num_discrete_coors = 128
)
autoencoder = autoencoder.to(device)
optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

for i in tqdm(range(epoch)):
    loss = autoencoder( #这里使用网络，是进去跑了forward
        vertices = planesur_vert,
        faces = planesur_face,
        face_edges = planesur_faceedge,
        in_em = in_em,
        GT = 1 #这里放真值
    )
    print('loss:',loss)
    optimizer.zero_grad()
    loss.backward() #这一步很花时间，但是没加optimizer是不是白给的
    optimizer.step()
torch.save(autoencoder.state_dict(), "/home/jxt/workspace/jxtnet/trytryweight.pt")
#读取：new_model.load_state_dict(torch.load(/home/jxt/workspace/jxtnet/trytryweight.pt))

print('训练结束时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
end_time0 = time.time()
print('训练用时：', time.strftime("%H:%M:%S", time.gmtime(end_time0-start_time0)))
#2024年4月2日22:25:07 终于从头到尾跟着跑完了一轮 明天开始魔改！
#2024年4月6日17:24:56 encoder和decoder加入了EM因素，NN魔改完成，接下来研究如何训练。
#2024年4月6日18:13:55 loss.backward optimizer .to(device)搞定，循环已开始，接下来研究如何dataloader
