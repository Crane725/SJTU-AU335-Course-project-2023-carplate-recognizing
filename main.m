clc
clear
close all%重置程序
% 1车牌图像预处理
% 1.1 RGB过滤和灰度极化
bar=waitbar(0,'Preprocessing in RGB filtering and bipolarizing...');
rgb_image=imread('test.jpg');%读取图片
fR=rgb_image(:, :, 1);%读取红色分量
fG=rgb_image(:, :, 2);%读取绿色分量
fB=rgb_image(:, :, 3);%读取蓝色分量

figure(1),subplot(1,2,1),
imshow(rgb_image);                                                                                                                                                                                                                                                                                                 
title('Original');

for i=1:size(rgb_image,1)   %对于图像中的像素颜色进行极化
    for j=1:size(rgb_image,2)
        if fR(i,j)>75|| fG(i,j)<30|| fG(i,j)>150|| fB(i,j)<120  %极化参数
            fR(i,j)=255;
            fG(i,j)=255;
            fB(i,j)=255;
        else
            fR(i,j)=0;
            fG(i,j)=0;
            fB(i,j)=0;
        end
    end
end
RGBtrans=cat(3,fR,fG,fB);
figure(1),subplot(1,2,2),
imshow(RGBtrans);                                                                                                                                                                                                                                                                                                 
title('RGB filtered & Bipolarized');
I=RGBtrans; %输入极化图像
I_gray=rgb2gray(I);

% 1.2车牌定位
waitbar(0,bar,'Preprocessing in plate locating...');
[y,x]=size(I_gray);		%将行列个数分别放入变量y,x中
Ysum=zeros(y,1);	%定义行列灰度值总和的向量
Xsum=zeros(x,1);
for i=1:y   %加和行列灰度值（单像素点归一化）
    for j=1:x
		Ysum(i)=Ysum(i)+double(I_gray(i,j));    %计算每行的像素和
        Xsum(j)=Xsum(j)+double(I_gray(i,j));    %计算每列的像素和
    end
end
[Ymin,Ypos]=min(Ysum);	%计算行像素值和最小值。
[Xmin,Xpos]=min(Xsum);  %计算列像素值和最小值
Ytop=Ypos;  %找出车牌的上边界
while((Ysum(Ytop)<=255*(x-5)) && (Ytop>1))
    Ytop=Ytop-1;
end

Ybottom=Ypos;   %找出车牌的下边界
while((Ysum(Ybottom)<=255*(x-5)) && (Ybottom<y))
    Ybottom=Ybottom+1;
end

Xleft=Xpos;  %找出车牌的左边界
while((Xsum(Xleft)<=255*(y-5)) && (Xleft>1))
    Xleft=Xleft-1;
end

Xright=Xpos;   %找出车牌的右边界
while((Xsum(Xright)<=255*(y-5)) && (Xright<x))
    Xright=Xright+1;
end



if 255*y-Xsum(Xleft+1)<=5000||255*x-Ysum(Ytop+1)<=3000
    filled=bwareaopen(I_gray,round(x*y/2));
    Ysum=zeros(y,1);	%定义行列灰度值总和的向量
    Xsum=zeros(x,1);
    for i=1:y   %加和行列灰度值（单像素点归一化）
        for j=1:x
		    Ysum(i)=Ysum(i)+double(filled(i,j))*255;    %计算每行的像素和
            Xsum(j)=Xsum(j)+double(filled(i,j))*255;    %计算每列的像素和
        end
    end
    Ysubtop=Ytop;
    Ysubbottom=Ybottom;
    Xsubleft=Xleft;
    Xsubright=Xright;
    for i=Ytop:Ybottom
        if Ysum(i)<=0.97*Ymin&&Ysubbottom==Ybottom
            Ysubbottom=Ybottom-i+Ytop;
        end
        if Ysum(i)<=0.97*Ymin&&Ysubtop==Ytop
            Ysubtop=i;
        end
        if Ysubbottom~=Ybottom&&Ysubtop~=Ytop
            break;
        end
    end
    for j=Xleft:Xright
        if Xsum(j)<=1.02*Xmin&&Xsubright==Xright
            Xsubright=Xright-j+Xleft;
        end
        if Xsum(j)<=1.02*Xmin&&Xsubleft==Xleft
            Xsubleft=j;
        end
        if Xsubright~=Xright&&Xsubleft~=Xleft
            break;
        end
    end
    if filled(Ytop,Xleft)==1&&filled(Ybottom,Xright)==1
        lt=[Xsubleft,Ysubtop];
        rt=[Xright,Ytop];
        lb=[Xleft,Ybottom];
        rb=[Xsubright,Ysubbottom];
    else
        if filled(Ybottom,Xleft)==1&&filled(Ytop,Xright)==1
            lt=[Xleft,Ytop];
            rt=[Xsubright,Ysubtop];
            lb=[Xsubleft,Ysubbottom];
            rb=[Xright,Ybottom];
        end
    end
    boxm=[lt;rt;lb;rb];
    boxf=[round(0.3175*x),round(0.5676*y);
    round(0.6867*x),round(0.5676*y);
    round(0.3175*x),round(0.7612*y);
    round(0.6867*x),round(0.7612*y)];
    TF = fitgeotrans(boxm,boxf,'projective');
    outview = imref2d(size(I_gray));
    I_gray = imwarp(I_gray,TF,'fillvalues',255,'outputview',outview);
    Ytop=round(0.5676*y);
    Ybottom=round(0.7612*y);
    Xleft=round(0.3175*x);
    Xright=round(0.6867*x);
end

Ytop=Ytop+20;
Ybottom=Ybottom-20;
Xleft=Xleft+20;
Xright=Xright-20;
dw=I_gray(Ytop:Ybottom,Xleft:Xright,:);
taka=Ybottom-Ytop+1;
yoko=Xright-Xleft+1;
Ytop=1;
Ybottom=taka;
Xleft=1;
Xright=yoko;
Ysum=zeros(taka,1);	%定义行列灰度值总和的向量
Xsum=zeros(yoko,1);
for i=1:taka   %加和行列灰度值（单像素点归一化）
    for j=1:yoko
		Ysum(i)=Ysum(i)+double(dw(i,j));    %计算每行的像素和
        Xsum(j)=Xsum(j)+double(dw(i,j));    %计算每列的像素和
    end
end
figure(2),subplot(1,2,1),
imshow(RGBtrans);                                                                                                                                                                                                                                                                                              
title('original');
figure(2),subplot(1,2,2),
imshow(dw);                                                                                                                                                                                                                                                                                                 
title('transformed & incised');
for i=1:taka
    if Ysum(i)<0.3*255*yoko&&Ysum(i)>0.17*255*yoko
        Ytop=i;
        break;
    end
end
for i=1:taka
    if Ysum(taka-i+1)<0.3*255*yoko&&Ysum(taka-i+1)>0.17*255*yoko
        Ybottom=taka-i+1;
        break;
    end
end
for j=1:yoko
    if Xsum(j)<0.2*255*taka
        Xleft=j;
        break;
    end
end
for j=1:yoko
    if Xsum(yoko-j+1)<0.2*255*taka
        Xright=yoko-j+1;
        break;
    end
end
dw=dw(Ytop:Ybottom,Xleft:Xright,:);
dw=imresize(dw,[200,811]);
se=strel('line',5,45);
dw=imerode(dw,se);
% 1.3 去噪

noiseinclude=dw;
taka=size(dw,1);
yoko=size(dw,2);
waitbar(0,bar,'  Preprocessing in noise eliminating...');
dw=imbinarize(dw);
%为图像添加白连通线，为后面删除小对象做准备
for i=1:taka    %为第一位汉字添加竖直白线以免删除小对象时被误删
    dw(i,round(0.08*yoko))=1;
    dw(i,round(0.06*yoko))=1;
    dw(i,round(0.03*yoko))=1;
end
for j=1:round(0.15*yoko)    %为第一位汉字添加水平白线以免删除小对象时被误删
    dw(round(0.5*taka),j)=1;
end

dw=bwareaopen(dw,round(8*taka)); %删除小对象操作
%根据邻域特征去掉为了删除小对象而添加的黑白线
for i=1:taka-1
    if dw(i+1,round(0.06*yoko))==1&&dw(i,round(0.06*yoko)-1)==0&&dw(i,round(0.06*yoko)+1)==0
        dw(i,round(0.06*yoko))=0;
    end
    if dw(i+1,round(0.08*yoko))==1&&dw(i,round(0.08*yoko)-1)==0&&dw(i,round(0.08*yoko)+1)==0
        dw(i,round(0.08*yoko))=0;
    end
    if dw(i+1,round(0.03*yoko))==1&&dw(i,round(0.03*yoko)-1)==0&&dw(i,round(0.03*yoko)+1)==0
        dw(i,round(0.03*yoko))=0;
    end
end
for j=1:round(0.15*yoko)
    dw(round(0.5*taka),j)=1;
    if dw(round(0.5*taka),j+1)==1&&dw(round(0.5*taka)+1,j)==0&&dw(round(0.5*taka)-1,j)==0
        dw(round(0.5*taka),j)=0;
    end
end
dw=bwareaopen(dw,100);    %再次删除小对象
dw=dw.*255;%将二值图像转为灰度值
figure(3),subplot(1,2,1),
imshow(noiseinclude);                                                                                                                                                                                                                                                                                              
title('original');
figure(3),subplot(1,2,2),
imshow(dw);                                                                                                                                                                                                                                                                                                 
title('noise eliminated');
                      

%进行字元分割
waitbar(0,bar,'Preprocessing in words incising...');
%定义列灰度值总和的向量
Xaccum=zeros(yoko,1);
Yaccum=zeros(taka,1);
for i=1:taka   %加和列灰度值（单像素点归一化）
    for j=1:yoko
        Yaccum(i)=Yaccum(i)+double(dw(i,j))/255;
    end
end
headline=1;
bottomline=taka;
for i=1:taka
    if (Yaccum(i)>=220)&&headline==1
        headline=i;
    end
    if (Yaccum(taka-i+1)>=220)&&bottomline==taka
        bottomline=taka-i+1;
    end
    if headline~=1&&bottomline~=taka
        break;
    end
end
dw=dw(headline:bottomline,:,:);
taka=bottomline-headline+1;
for i=1:taka   %加和列灰度值（单像素点归一化）
    for j=1:yoko
        Xaccum(j)=Xaccum(j)+double(dw(i,j))/255;
    end
end

% 初始化必要参数
hori=zeros(14,1);
tmp=1;
last=1;     %用last来记录上次裁剪的方位
for j=1:yoko-3
    if tmp==15
            break;
    end
    if  Xaccum(j)~=0&&Xaccum(j+1)~=0&&Xaccum(j+2)~=0&&Xaccum(j+3)~=0&&last==1
            hori(tmp)=j;
            last=last*-1;
            tmp=tmp+1;
            continue;
    end
    if  Xaccum(j)==0&&Xaccum(j+1)==0&&Xaccum(j+2)==0&&Xaccum(j+3)==0&&last==-1&&j-hori(tmp-1)>=70
            hori(tmp)=j;
            last=last*-1;
            tmp=tmp+1;
    end
    if j==yoko-3&&hori(14)==0
        hori(14)=yoko;
    end
end
file_path='.\samples\';
plate=['i','i','i','i','i','i','i'];
class=['location1\';'location2\';'sequences\';'sequences\';'sequences\';'sequences\';'sequences\'];
error=0;
for chara=1:7
    if error~=0
        break;
    end
    score=-inf;
    wordchara=dw(:,round(hori(2*chara-1)):round(hori(2*chara)),:);
    figure(4),subplot(2,7,chara),imshow(wordchara);
    for k=3:size(dir([file_path,class(chara,:)]))
        processing=sprintf('Words recognizing ... %.1f %s',100*(k-2+31*(chara>1)+24*(chara>2)+34*(chara-3)*(chara>3))/225,'%');
        waitbar(((k-2+31*(chara>1)+24*(chara>2)+34*(chara-3)*(chara>3)))/225,bar,processing);
        file=dir([file_path,class(chara,:)]);
        file_name=file(k).name;
        sample=imread([[file_path,class(chara,:)],file_name]);
        sample=imresize(sample,[size(wordchara,1),size(wordchara,2)]);
        tmpscore=0;
        for i=1:size(sample,1)   %对于图像中的像素颜色进行极化
            for j=1:size(sample,2)
                if sample(i,j,1)>127  %极化参数
                    sample(i,j,1)=255;
                else
                    sample(i,j,1)=0;
                end
            end
        end
        for i=1:size(wordchara,1)
            for j=1:size(wordchara,2)
                if sample(i,j)==wordchara(i,j)&&wordchara(i,j)==255
                    tmpscore=tmpscore+1;
                else
                    if sample(i,j)==wordchara(i,j)&&wordchara(i,j)==0
                        tmpscore=tmpscore+1;
                    else
                        tmpscore=tmpscore-1;
                    end
                end
            end
        end
        if tmpscore>score
            score=tmpscore;
            tmpplate=file_name;
        else
            if tmpscore==score
                error=chara;
                break;
            else
                continue;
            end
        end
    end
    figure(4),subplot(2,7,chara+7),imshow(imread([[file_path,class(chara,:)],tmpplate]));
    title(erase(tmpplate,".jpg"),'FontSize',20,'Color',[1,1,1],'EdgeColor',[100/255,150/255,170/255],'BackgroundColor',[0,80/255,170/255],'FontWeight','bold');
    tmpplate=erase(tmpplate,".jpg");
    plate(chara)=tmpplate;
end
if error==0
    waitbar(1,bar,'Running Over');
    plate=[plate(1:2),'·',plate(3:7)];
    sprintf(plate)
else
    waitbar(0,bar,sprintf('Equivalent ERROR occurred in word %d',error));
end